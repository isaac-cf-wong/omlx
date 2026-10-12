# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen3.5-family prefill stack on DFlash targets."""

import logging
from types import SimpleNamespace

import pytest

from omlx.model_settings import ModelSettings
from omlx.patches import dflash_qwen35_accel as accel
from omlx.patches import qwen35_ane_prefill

_TUNED = dict(
    qwen35_ane_prefill_enabled=True,
    qwen35_ane_prefill_sequence_length=1024,
    qwen35_ane_prefill_tail_padding_min_tokens=782,
    qwen35_ane_prefill_fraction=0.6,
    qwen35_ane_prefill_max_layers=40,
    qwen35_ane_prefill_gdn=True,
    qwen35_ane_prefill_gdn_fraction=0.375,
    qwen35_ane_prefill_gdn_max_layers=30,
    qwen35_ane_prefill_dual_ane=False,
    qwen35_ane_prefill_cpu_enabled=True,
    qwen35_ane_prefill_cpu_fraction=0.1,
    qwen35_ane_prefill_cpu_down_fraction=0.2,
    qwen35_ane_prefill_cpu_gdn_fraction=0.05,
    qwen35_ane_prefill_cpu_threads=6,
    qwen35_ane_prefill_cpu_shared_resource=False,
)

_TUNED_KWARGS = {
    "sequence_length": 1024,
    "tail_padding_min_tokens": 782,
    "fraction": 0.6,
    "max_layers": 40,
    "gdn": True,
    "gdn_fraction": 0.375,
    "gdn_max_layers": 30,
    "dual_ane": False,
    "ane_down_fraction": 0.0,
    "fused_down": False,
    "cpu_fraction": 0.1,
    "cpu_down_fraction": 0.2,
    "cpu_gdn_fraction": 0.05,
    "cpu_threads": 6,
    "cpu_shared_resource": False,
}


class TestAnePrefillKwargs:
    def test_tuned_profile_maps_every_setting(self):
        kwargs = qwen35_ane_prefill.qwen35_ane_prefill_kwargs(
            ModelSettings(**_TUNED), "qwen3_5"
        )
        assert kwargs == _TUNED_KWARGS

    def test_cpu_fractions_zero_while_cpu_sharing_is_off(self):
        settings = ModelSettings(**{**_TUNED, "qwen35_ane_prefill_cpu_enabled": False})
        kwargs = qwen35_ane_prefill.qwen35_ane_prefill_kwargs(settings, "qwen3_5")
        assert kwargs["cpu_fraction"] == 0.0
        assert kwargs["cpu_down_fraction"] == 0.0
        assert kwargs["cpu_gdn_fraction"] == 0.0

    def test_fused_down_sends_the_mlp_fraction_to_the_down_projection(self):
        settings = ModelSettings(**{**_TUNED, "qwen35_ane_prefill_fused_down": True})
        kwargs = qwen35_ane_prefill.qwen35_ane_prefill_kwargs(settings, "qwen3_5")
        assert kwargs["fused_down"] is True
        assert kwargs["ane_down_fraction"] == 0.6

    def test_unset_fraction_takes_the_backend_default(self):
        kwargs = qwen35_ane_prefill.qwen35_ane_prefill_kwargs(
            ModelSettings(), "qwen3_5"
        )
        assert kwargs["fraction"] == 0.53
        assert kwargs["sequence_length"] == 2048
        assert kwargs["tail_padding_min_tokens"] == 0

    def test_kwargs_are_accepted_by_enable(self):
        import inspect

        params = inspect.signature(
            qwen35_ane_prefill.enable_qwen35_ane_prefill
        ).parameters
        assert set(_TUNED_KWARGS) <= set(params)


@pytest.fixture
def captured_enable(monkeypatch):
    calls = []

    def fake_enable(model, **kwargs):
        calls.append((model, kwargs))
        model._omlx_ane_gdn_prefill_count = 3
        return 5

    monkeypatch.setattr(qwen35_ane_prefill, "enable_qwen35_ane_prefill", fake_enable)
    return calls


class TestEnableDflashAne:
    def test_tuned_profile_reaches_the_dflash_target(self, captured_enable):
        model = SimpleNamespace()
        count = accel.enable_dflash_qwen35_ane(
            model, ModelSettings(**_TUNED), "qwen3_5", prefill_step_size=2048
        )
        assert count == 5
        assert captured_enable == [(model, _TUNED_KWARGS)]

    def test_off_by_default(self, captured_enable):
        count = accel.enable_dflash_qwen35_ane(
            SimpleNamespace(), ModelSettings(), "qwen3_5", prefill_step_size=2048
        )
        assert count == 0
        assert captured_enable == []

    def test_non_qwen_target_is_skipped(self, captured_enable):
        count = accel.enable_dflash_qwen35_ane(
            SimpleNamespace(), ModelSettings(**_TUNED), "gemma4", prefill_step_size=2048
        )
        assert count == 0
        assert captured_enable == []

    def test_invalid_settings_keep_prefill_on_gpu(self, captured_enable, caplog):
        settings = ModelSettings(
            **{**_TUNED, "qwen35_ane_prefill_sequence_length": 1000}
        )
        with caplog.at_level(logging.WARNING):
            count = accel.enable_dflash_qwen35_ane(
                SimpleNamespace(), settings, "qwen3_5", prefill_step_size=2048
            )
        assert count == 0
        assert captured_enable == []
        assert "not enabled for DFlash" in caplog.text

    def test_compile_failure_keeps_prefill_on_gpu(self, monkeypatch, caplog):
        def broken(model, **kwargs):
            raise RuntimeError("ANE compile failed")

        monkeypatch.setattr(qwen35_ane_prefill, "enable_qwen35_ane_prefill", broken)
        with caplog.at_level(logging.WARNING):
            count = accel.enable_dflash_qwen35_ane(
                SimpleNamespace(),
                ModelSettings(**_TUNED),
                "qwen3_5",
                prefill_step_size=2048,
            )
        assert count == 0
        assert "not enabled for DFlash" in caplog.text

    @pytest.mark.parametrize(
        ("sequence_length", "warns"), [(2048, False), (3072, True)]
    )
    def test_warns_when_the_tile_exceeds_the_dflash_step(
        self, captured_enable, caplog, sequence_length, warns
    ):
        settings = ModelSettings(
            **{**_TUNED, "qwen35_ane_prefill_sequence_length": sequence_length}
        )
        with caplog.at_level(logging.WARNING):
            accel.enable_dflash_qwen35_ane(
                SimpleNamespace(), settings, "qwen3_5", prefill_step_size=2048
            )
        assert ("exceeds the DFlash prefill step" in caplog.text) is warns


class TestClassPatches:
    @pytest.fixture
    def installers(self, monkeypatch):
        from omlx.patches import (
            qwen35_fa256_attention,
            qwen35_q4_mlp,
            sdpa256_attention,
        )

        order = []

        def record(name):
            def install(*args, **kwargs):
                order.append(name)
                return True

            return install

        monkeypatch.setattr(
            sdpa256_attention, "apply_sdpa256_attention_patch", record("sdpa256")
        )
        monkeypatch.setattr(
            qwen35_fa256_attention,
            "apply_qwen35_fa256_attention_patch",
            record("fa256"),
        )
        monkeypatch.setattr(
            qwen35_q4_mlp, "apply_qwen35_q4_mlp_patch", record("q4_mlp")
        )
        monkeypatch.setattr(
            qwen35_q4_mlp,
            "apply_qwen35_q4_lm_prefill_linear_patch",
            record("q4_lm_prefill_linear"),
        )
        return order

    def test_installs_in_the_batched_engine_order(self, installers):
        installed = accel.install_dflash_qwen35_class_patches(ModelSettings())
        assert installers == ["sdpa256", "fa256", "q4_mlp", "q4_lm_prefill_linear"]
        assert all(installed.values())

    @pytest.mark.parametrize(
        ("setting", "skipped"),
        [
            ("sdpa256_prefill_enabled", ["sdpa256"]),
            ("fa256_steel_prefill_enabled", ["fa256"]),
            ("qwen35_q4_mlp_prefill_enabled", ["q4_mlp", "q4_lm_prefill_linear"]),
        ],
    )
    def test_each_setting_disables_its_patch(self, installers, setting, skipped):
        # Not ModelSettings fields: like the batched engines, absent means on.
        accel.install_dflash_qwen35_class_patches(SimpleNamespace(**{setting: False}))
        expected = ["sdpa256", "fa256", "q4_mlp", "q4_lm_prefill_linear"]
        assert installers == [name for name in expected if name not in skipped]

    def test_a_failing_patch_does_not_stop_the_rest(self, installers, monkeypatch):
        from omlx.patches import qwen35_fa256_attention

        def broken():
            raise RuntimeError("no kernel")

        monkeypatch.setattr(
            qwen35_fa256_attention, "apply_qwen35_fa256_attention_patch", broken
        )
        installed = accel.install_dflash_qwen35_class_patches(ModelSettings())
        assert installers == ["sdpa256", "q4_mlp", "q4_lm_prefill_linear"]
        assert installed["fa256"] is False


def test_release_frees_the_target_banks(monkeypatch):
    released = []
    monkeypatch.setattr(
        qwen35_ane_prefill,
        "release_qwen35_ane_prefill",
        lambda model: released.append(model) or (4, 8),
    )
    model = SimpleNamespace()
    accel.release_dflash_qwen35_ane(model)
    accel.release_dflash_qwen35_ane(None)
    assert released == [model]


@pytest.mark.asyncio
async def test_engine_installs_before_load_and_releases_on_stop(monkeypatch):
    try:
        from dflash_mlx.cache import manager as cache_manager
        from dflash_mlx.runtime import loading as dflash_loading

        from omlx.engine import dflash as dflash_mod
        from omlx.engine.dflash import DFlashEngine
        from omlx.patches import dflash_lifecycle
    except ImportError:
        pytest.skip("dflash-mlx not installed")

    events = []
    target = SimpleNamespace()

    def fake_load_target_bundle(model_ref, **kwargs):
        events.append("load_target")
        return SimpleNamespace(
            model=target,
            tokenizer=SimpleNamespace(
                name_or_path="fake-target", eos_token_id=1, eos_token_ids=[1]
            ),
            meta={"config": {"model_type": "qwen3_5"}},
            target_ops=SimpleNamespace(),
        )

    class FakeDraft:
        def bind_target_model(self, target_model, *, target_ops):
            pass

    monkeypatch.setattr(dflash_loading, "load_target_bundle", fake_load_target_bundle)
    monkeypatch.setattr(
        dflash_loading, "load_draft_bundle", lambda *a, **k: (FakeDraft(), {})
    )
    monkeypatch.setattr(
        dflash_mod, "maybe_apply_pre_load_patches", lambda *a, **k: None
    )
    monkeypatch.setattr(dflash_lifecycle, "install_dflash_lifecycle_wrap", lambda: True)
    monkeypatch.setattr(
        dflash_mod, "load_generation_config_token_ids", lambda *a, **k: set()
    )
    monkeypatch.setattr(dflash_mod, "detect_output_parser", lambda *a, **k: None)
    monkeypatch.setattr(dflash_mod, "set_model_info_from_model", lambda *a: None)
    monkeypatch.setattr(cache_manager, "shutdown_runtime_cache_manager", lambda: None)
    monkeypatch.setattr(
        accel,
        "install_dflash_qwen35_class_patches",
        lambda settings: events.append("class_patches") or {},
    )

    def fake_enable(model, settings, model_type, *, prefill_step_size):
        events.append(("ane", model, model_type, prefill_step_size))
        return 5

    monkeypatch.setattr(accel, "enable_dflash_qwen35_ane", fake_enable)
    monkeypatch.setattr(
        accel,
        "release_dflash_qwen35_ane",
        lambda model: events.append(("release", model)),
    )

    engine = DFlashEngine(
        model_name="test-model",
        draft_model_path="test-draft",
        model_settings=ModelSettings(**_TUNED),
    )
    await engine.start()
    await engine.stop()

    assert events == [
        "class_patches",
        "load_target",
        ("ane", target, "qwen3_5", 2048),
        ("release", target),
    ]
