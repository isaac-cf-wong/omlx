# SPDX-License-Identifier: Apache-2.0
"""NVIDIA Nemotron 3.5 Bolt (``model_type: nemotron_h_puzzle``).

The architecture is mlx-lm's ``nemotron_h`` (Mamba2 / MoE / GQA hybrid with
latent MoE projections) with two differences:

* ``block_configs`` sets the routed expert count, expert width and top-k per
  MoE layer instead of once for the model.
* Checkpoints mix three weight formats, chosen per module by the config's
  ``quantization`` entries:

  - ``iq2_xxs`` routed and shared experts (``omlx.custom_kernels.iq2_xxs``);
  - ``nvfp4`` Mamba in/out projections with a per-tensor ``global_scale``,
    which ``nn.QuantizedLinear`` cannot hold;
  - 8-bit ``affine`` everywhere else, stored with float16 scales. Under
    bfloat16 activations MLX promotes those outputs to float32, so the
    layer casts them back to the input dtype.

The modules for the first two formats are built here already quantized;
their ``to_quantized`` returns the module unchanged so mlx-lm's
``nn.quantize`` pass leaves them alone.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import nemotron_h as base

from omlx.custom_kernels.iq2_xxs import fast as iq2

# ModelOpt's per-tensor NVFP4 scale is global_scale / (448 * 6).
_NVFP4_GLOBAL_DIVISOR = 2688.0
# Rows of x from which NVFP4 layers dequantize and use the dense matmul.
_NVFP4_DEQUANT_MIN_ROWS = 1024


@dataclass()
class ModelArgs(base.ModelArgs):
    block_configs: list[dict] | None = None
    quantization: dict | None = None

    def __post_init__(self):
        # transformers writes non-finite floats as {"__float__": "Infinity"}.
        if self.time_step_limit is not None:
            self.time_step_limit = tuple(
                float(v["__float__"]) if isinstance(v, dict) else float(v)
                for v in self.time_step_limit
            )
        super().__post_init__()


class IQ2XXSLinear(nn.Module):
    def __init__(self, input_dims: int, output_dims: int):
        super().__init__()
        self.weight = mx.zeros((output_dims, input_dims // 16), dtype=mx.uint32)
        self.scales = mx.zeros((output_dims, input_dims // 256), dtype=mx.float16)

    def __call__(self, x: mx.array) -> mx.array:
        return iq2.iq2_xxs_matmul(x, self.weight, self.scales)

    def to_quantized(self, *args, **kwargs):
        return self


class IQ2XXSSwitchLinear(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, num_experts: int):
        super().__init__()
        self.weight = mx.zeros(
            (num_experts, output_dims, input_dims // 16), dtype=mx.uint32
        )
        self.scales = mx.zeros(
            (num_experts, output_dims, input_dims // 256), dtype=mx.float16
        )

    def __call__(
        self, x: mx.array, indices: mx.array, activation: str | None = None
    ) -> mx.array:
        return iq2.gather_iq2_xxs_matmul(
            x, self.weight, self.scales, indices, activation=activation
        )

    def to_quantized(self, *args, **kwargs):
        return self


class IQ2XXSSwitchMLP(nn.Module):
    """Non-gated relu^2 experts, the IQ2 counterpart of ``SwitchMLP``."""

    def __init__(self, input_dims: int, hidden_dims: int, num_experts: int):
        super().__init__()
        self.fc1 = IQ2XXSSwitchLinear(input_dims, hidden_dims, num_experts)
        self.fc2 = IQ2XXSSwitchLinear(hidden_dims, input_dims, num_experts)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        if indices.size >= iq2.GATHER_QMM_MIN_ROWS:
            return self._sorted(x, indices)
        x = mx.expand_dims(x, (-2, -3))
        x = self.fc1(x, indices, activation="relu2")
        x = self.fc2(x, indices)
        return x.squeeze(-2)

    def _sorted(self, x: mx.array, indices: mx.array) -> mx.array:
        # Sort assignments by expert once; fc1's output stays in that order
        # as fc2's input, so only the final result is unsorted.
        k = indices.shape[-1]
        flat = indices.reshape(-1).astype(mx.uint32)
        order = mx.argsort(flat)
        experts = flat[order]
        x2 = x.reshape(-1, x.shape[-1])
        h = iq2.gather_qmm_sorted(
            x2,
            self.fc1.weight,
            self.fc1.scales,
            order // k,
            experts,
            activation="relu2",
        )
        y = iq2.gather_qmm_sorted(
            h,
            self.fc2.weight,
            self.fc2.scales,
            mx.arange(flat.size, dtype=mx.uint32),
            experts,
        )
        return y[mx.argsort(order)].reshape(*indices.shape, -1)


@partial(mx.compile, shapeless=True)
def _apply_global_scale(y: mx.array, global_scale: mx.array) -> mx.array:
    # Scale in float32: the factor is ~1e-4 and loses ~0.4% as bfloat16.
    # Compiled so the multiply and the cast are one kernel.
    return (y * (global_scale / _NVFP4_GLOBAL_DIVISOR)).astype(y.dtype)


class NVFP4Linear(nn.Module):
    def __init__(self, input_dims: int, output_dims: int):
        super().__init__()
        self.weight = mx.zeros((output_dims, input_dims // 8), dtype=mx.uint32)
        self.scales = mx.zeros((output_dims, input_dims // 16), dtype=mx.uint8)
        self.global_scale = mx.ones((), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        if x.size // x.shape[-1] >= _NVFP4_DEQUANT_MIN_ROWS:
            # Prefill: MLX's dense GEMM outruns its nvfp4 qmm. Without the
            # global scale each weight (e2m1 * e4m3, at most 6 significant
            # bits) is exact in bfloat16, so only the scaling order changes.
            w = mx.dequantize(
                self.weight,
                self.scales,
                group_size=16,
                bits=4,
                mode="nvfp4",
                dtype=x.dtype,
            )
            y = x @ w.T
        else:
            y = mx.quantized_matmul(
                x,
                self.weight,
                self.scales,
                transpose=True,
                group_size=16,
                bits=4,
                mode="nvfp4",
            )
        return _apply_global_scale(y, self.global_scale)

    def to_quantized(self, *args, **kwargs):
        return self


class _CastQuantizedLinear(nn.QuantizedLinear):
    def __call__(self, x: mx.array) -> mx.array:
        return super().__call__(x).astype(x.dtype)


class AffineLinear(nn.Linear):
    """``nn.Linear`` that quantizes to a dtype-preserving ``QuantizedLinear``."""

    def to_quantized(
        self, group_size=None, bits=None, mode="affine", quantize_input=False
    ):
        if quantize_input:
            raise ValueError("activation quantization is not supported here")
        out_dims, in_dims = self.weight.shape
        return _CastQuantizedLinear(
            in_dims,
            out_dims,
            bias="bias" in self,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )


def _mode(args: ModelArgs, path: str) -> str | None:
    quant = args.quantization
    if not quant:
        return None
    entry = quant.get(path)
    if isinstance(entry, dict):
        return entry.get("mode", "affine")
    return quant.get("mode", "affine")


def _linear(args: ModelArgs, path: str, input_dims: int, output_dims: int):
    mode = _mode(args, path)
    if mode == "iq2_xxs":
        return IQ2XXSLinear(input_dims, output_dims)
    if mode == "nvfp4":
        return NVFP4Linear(input_dims, output_dims)
    return AffineLinear(input_dims, output_dims, bias=False)


def _layer_args(args: ModelArgs, block: dict) -> ModelArgs:
    overrides = {
        k: block[k]
        for k in ("n_routed_experts", "moe_intermediate_size", "num_experts_per_tok")
        if k in block
    }
    return replace(args, **overrides) if overrides else args


def _build_block(args: ModelArgs, i: int, block_type: str) -> base.NemotronHBlock:
    block = base.NemotronHBlock(args, block_type)
    prefix = f"backbone.layers.{i}.mixer"
    mixer = block.mixer
    h = args.hidden_size
    if block_type == "M":
        mixer.in_proj = _linear(
            args, f"{prefix}.in_proj", h, mixer.in_proj.weight.shape[0]
        )
        mixer.out_proj = _linear(args, f"{prefix}.out_proj", mixer.intermediate_size, h)
    elif block_type == "*":
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            out_dims, in_dims = getattr(mixer, name).weight.shape
            setattr(mixer, name, _linear(args, f"{prefix}.{name}", in_dims, out_dims))
    elif block_type == "E":
        latent = args.moe_latent_size or h
        if _mode(args, f"{prefix}.switch_mlp.fc1") == "iq2_xxs":
            mixer.switch_mlp = IQ2XXSSwitchMLP(
                latent, args.moe_intermediate_size, args.n_routed_experts
            )
        if args.n_shared_experts is not None:
            shared = mixer.shared_experts
            inter = args.moe_shared_expert_intermediate_size
            shared.up_proj = _linear(args, f"{prefix}.shared_experts.up_proj", h, inter)
            shared.down_proj = _linear(
                args, f"{prefix}.shared_experts.down_proj", inter, h
            )
        if args.moe_latent_size is not None:
            mixer.fc1_latent_proj = _linear(
                args, f"{prefix}.fc1_latent_proj", h, latent
            )
            mixer.fc2_latent_proj = _linear(
                args, f"{prefix}.fc2_latent_proj", latent, h
            )
    return block


class NemotronHPuzzleModel(base.NemotronHModel):
    def __init__(self, args: ModelArgs):
        nn.Module.__init__(self)
        configs = args.block_configs or [{} for _ in args.hybrid_override_pattern]
        if len(configs) != len(args.hybrid_override_pattern):
            raise ValueError(
                f"{len(configs)} block_configs for "
                f"{len(args.hybrid_override_pattern)} layers"
            )
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            _build_block(_layer_args(args, cfg), i, block_type)
            for i, (block_type, cfg) in enumerate(
                zip(args.hybrid_override_pattern, configs)
            )
        ]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        pattern = args.hybrid_override_pattern
        # Cache slots that carry the attention and SSM masks (see base).
        self.fa_idx = 0
        self.ssm_idx = 0
        for b in pattern:
            if b == "*":
                break
            elif b == "M":
                self.fa_idx += 1
        for b in pattern:
            if b == "*":
                self.ssm_idx += 1
            elif b == "M":
                break


class Model(base.Model):
    def __init__(self, args: ModelArgs):
        nn.Module.__init__(self)
        self.args = args
        self.backbone = NemotronHPuzzleModel(args)
        self.lm_head = _linear(args, "lm_head", args.hidden_size, args.vocab_size)
        self.model_type = args.model_type

    def sanitize(self, weights):
        weights = {k: v for k, v in weights.items() if not k.startswith("mtp.")}
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
        return weights

    @property
    def cast_predicate(self):
        def predicate(k):
            return not any(
                s in k
                for s in ("e_score_correction_bias", "A_log", "scales", "global_scale")
            )

        return predicate


__all__ = ["Model", "ModelArgs"]
