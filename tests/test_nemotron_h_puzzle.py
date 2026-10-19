# SPDX-License-Identifier: Apache-2.0
"""Tests for the Nemotron 3.5 Bolt (``nemotron_h_puzzle``) model patch."""

from __future__ import annotations

import importlib
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from omlx.custom_kernels.iq2_xxs import fast as iq2
from omlx.patches.nemotron_h_puzzle import apply_nemotron_h_puzzle_patch

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="iq2_xxs kernels need Metal"
)

apply_nemotron_h_puzzle_patch()
puzzle = importlib.import_module("mlx_lm.models.nemotron_h_puzzle")

_BLOCKS = [
    {"block_type": "mamba"},
    {
        "block_type": "moe",
        "moe_intermediate_size": 256,
        "n_routed_experts": 6,
        "num_experts_per_tok": 2,
    },
    {"block_type": "attention"},
    {
        "block_type": "moe",
        "moe_intermediate_size": 512,
        "n_routed_experts": 10,
        "num_experts_per_tok": 3,
    },
]


def _config() -> dict:
    quant = {"mode": "affine", "group_size": 32, "bits": 8}
    for i, b in enumerate(_BLOCKS):
        p = f"backbone.layers.{i}.mixer"
        if b["block_type"] == "mamba":
            for m in ("in_proj", "out_proj"):
                quant[f"{p}.{m}"] = {"mode": "nvfp4", "group_size": 16, "bits": 4}
        if b["block_type"] == "moe":
            for m in (
                "switch_mlp.fc1",
                "switch_mlp.fc2",
                "shared_experts.up_proj",
                "shared_experts.down_proj",
            ):
                quant[f"{p}.{m}"] = {"mode": "iq2_xxs", "group_size": 256, "bits": 2}
    return {
        "model_type": "nemotron_h_puzzle",
        "vocab_size": 64,
        "hidden_size": 256,
        "intermediate_size": 512,
        "max_position_embeddings": 1024,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "attention_bias": False,
        "mamba_num_heads": 4,
        "mamba_head_dim": 64,
        "mamba_proj_bias": False,
        "ssm_state_size": 16,
        "conv_kernel": 4,
        "n_groups": 2,
        "mlp_bias": False,
        "layer_norm_epsilon": 1e-5,
        "use_bias": False,
        "use_conv_bias": True,
        "layers_block_type": [b["block_type"] for b in _BLOCKS],
        "block_configs": _BLOCKS,
        "moe_latent_size": 256,
        "moe_shared_expert_intermediate_size": 512,
        "n_shared_experts": 1,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.0,
        "time_step_limit": [0.0, {"__float__": "Infinity"}],
        "quantization": quant,
    }


def _build(seed: int = 0):
    config = _config()
    model = puzzle.Model(puzzle.ModelArgs.from_dict(config))
    quant = config["quantization"]
    # As mlx-lm's load_model: explicit entries win, other Linears take the
    # default, everything else (embeddings, gate) stays unquantized.
    nn.quantize(
        model,
        group_size=32,
        bits=8,
        class_predicate=lambda p, m: quant.get(p, isinstance(m, nn.Linear)),
    )
    rng = np.random.default_rng(seed)

    def normal(shape, std=0.05):
        return mx.array(rng.standard_normal(shape).astype(np.float32) * std)

    weights = {}
    for name, m in model.named_modules():
        if isinstance(m, (puzzle.IQ2XXSLinear, puzzle.IQ2XXSSwitchLinear)):
            codes = rng.integers(0, 2**32, size=m.weight.shape, dtype=np.uint64)
            weights[f"{name}.weight"] = mx.array(codes.astype(np.uint32))
            weights[f"{name}.scales"] = mx.array(
                (rng.uniform(size=m.scales.shape) * 2e-3).astype(np.float16)
            )
        elif isinstance(m, puzzle.NVFP4Linear):
            out_dims, in_dims = m.weight.shape[0], m.weight.shape[1] * 8
            wq, sq = mx.quantize(
                normal((out_dims, in_dims)), group_size=16, bits=4, mode="nvfp4"
            )
            weights[f"{name}.weight"], weights[f"{name}.scales"] = wq, sq
            weights[f"{name}.global_scale"] = mx.array(2688.0 * 0.5)
        elif isinstance(m, nn.QuantizedLinear):
            out_dims, in_dims = m.weight.shape[0], m.weight.shape[1] * 4
            wq, sq, bq = mx.quantize(normal((out_dims, in_dims)), group_size=32, bits=8)
            # float32 scales: with the checkpoint's float16 scales MLX rebuilds
            # these weights at float16 precision (~2e-4 per layer), which would
            # mask the wiring errors the tests below look for.
            weights[f"{name}.weight"] = wq
            weights[f"{name}.scales"] = sq
            weights[f"{name}.biases"] = bq
    for name, param in tree_flatten(model.parameters()):
        if name in weights:
            continue
        if "A_log" in name:
            weights[name] = mx.zeros(param.shape)
        elif "norm" in name or name.endswith(".D"):
            weights[name] = mx.ones(param.shape)
        else:
            weights[name] = normal(param.shape)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model


def _np(a) -> np.ndarray:
    return np.array(a.astype(mx.float32), dtype=np.float64)


def test_model_args_decode_json_infinity():
    args = puzzle.ModelArgs.from_dict(_config())
    assert args.time_step_limit == (0.0, float("inf"))


def test_layers_follow_block_configs():
    model = _build()
    layers = model.backbone.layers
    assert isinstance(layers[0].mixer.in_proj, puzzle.NVFP4Linear)
    for i, block in ((1, _BLOCKS[1]), (3, _BLOCKS[3])):
        mlp = layers[i].mixer.switch_mlp
        assert isinstance(mlp, puzzle.IQ2XXSSwitchMLP)
        assert mlp.fc1.weight.shape == (
            block["n_routed_experts"],
            block["moe_intermediate_size"],
            256 // 16,
        )
        assert layers[i].mixer.gate.top_k == block["num_experts_per_tok"]
        assert isinstance(layers[i].mixer.shared_experts.up_proj, puzzle.IQ2XXSLinear)
    assert isinstance(layers[2].mixer.q_proj, nn.QuantizedLinear)
    assert isinstance(model.lm_head, nn.QuantizedLinear)


@pytest.mark.parametrize("tokens", [1, 20, 40])  # qmv, qmv, sorted-qmm paths
def test_moe_layer_matches_dense_reference(tokens):
    model = _build()
    moe = model.backbone.layers[3].mixer
    x = mx.array(
        np.random.default_rng(tokens)
        .standard_normal((1, tokens, 256))
        .astype(np.float32)
    )
    inds, scores = moe.gate(x)
    out = _np(moe(x))

    def dense(q):
        return np.array(
            mx.dequantize(
                q.weight, q.scales, q.biases, group_size=32, bits=8, dtype=mx.float32
            ),
            dtype=np.float64,
        )

    w1 = _np(
        iq2.dequantize(moe.switch_mlp.fc1.weight, moe.switch_mlp.fc1.scales, mx.float32)
    )
    w2 = _np(
        iq2.dequantize(moe.switch_mlp.fc2.weight, moe.switch_mlp.fc2.scales, mx.float32)
    )
    up = _np(
        iq2.dequantize(
            moe.shared_experts.up_proj.weight,
            moe.shared_experts.up_proj.scales,
            mx.float32,
        )
    )
    down = _np(
        iq2.dequantize(
            moe.shared_experts.down_proj.weight,
            moe.shared_experts.down_proj.scales,
            mx.float32,
        )
    )
    xn = _np(x)[0]
    lat = xn @ dense(moe.fc1_latent_proj).T
    idx, sc = np.array(inds)[0], _np(scores)[0]
    routed = np.zeros_like(lat)
    for t in range(tokens):
        for j in range(idx.shape[1]):
            h = np.maximum(w1[idx[t, j]] @ lat[t], 0.0) ** 2
            routed[t] += sc[t, j] * (w2[idx[t, j]] @ h)
    ref = (
        routed @ dense(moe.fc2_latent_proj).T
        + (np.maximum(xn @ up.T, 0.0) ** 2) @ down.T
    )
    assert np.abs(out[0] - ref).max() / np.abs(ref).max() < 1e-4


def test_nvfp4_linear_applies_global_scale_on_both_paths():
    lin = puzzle.NVFP4Linear(256, 48)
    w = mx.array(np.random.default_rng(0).standard_normal((48, 256)).astype(np.float32))
    lin.weight, lin.scales = mx.quantize(w, group_size=16, bits=4, mode="nvfp4")
    lin.global_scale = mx.array(0.359375)
    deq = _np(
        mx.dequantize(lin.weight, lin.scales, group_size=16, bits=4, mode="nvfp4")
    )
    for rows in (3, puzzle._NVFP4_DEQUANT_MIN_ROWS):
        x = np.random.default_rng(rows).standard_normal((rows, 256)).astype(np.float32)
        ref = x @ deq.T * (0.359375 / 2688.0)
        out = _np(lin(mx.array(x)))
        assert np.abs(out - ref).max() / np.abs(ref).max() < 1e-5


def test_affine_layers_keep_activation_dtype():
    model = _build()
    x = mx.ones((1, 2, 256), dtype=mx.bfloat16)
    assert model.backbone.layers[2].mixer.q_proj(x).dtype == mx.bfloat16


def test_incremental_decode_matches_full_forward():
    model = _build()
    toks = mx.array(np.random.default_rng(3).integers(0, 64, size=(1, 12)))
    full = _np(model(toks))[0, -1]
    cache = model.make_cache()
    model(toks[:, :-1], cache=cache)
    step = _np(model(toks[:, -1:], cache=cache))[0, -1]
    assert np.isfinite(full).all()
    # mlx-lm's nemotron_h prefill and single-step SSM paths disagree by ~4e-3
    # even unquantized in float32; a broken cache is off by O(1).
    assert np.abs(step - full).max() / np.abs(full).max() < 2e-2


def test_pre_load_patch_wiring(tmp_path, monkeypatch):
    from omlx.utils import model_loading

    calls = []
    monkeypatch.setattr(
        "omlx.patches.nemotron_h_puzzle.apply_nemotron_h_puzzle_patch",
        lambda: calls.append(True) or True,
    )
    (tmp_path / "config.json").write_text('{"model_type": "nemotron_h_puzzle"}')
    model_loading.maybe_apply_pre_load_patches(str(tmp_path))
    assert calls == [True]
    assert "mlx_lm.models.nemotron_h_puzzle" in sys.modules
