# SPDX-License-Identifier: Apache-2.0
"""Tests for overlap-tail coverage of the ANE prefill residual.

A wide prefill splits into fixed-shape ANE tiles plus a residual tail. With the
overlap enabled the tail is covered by re-running the last ``sequence_length``
rows and keeping only its last ``tail_rows`` outputs, which is only correct if
the slice lands exactly where the tiles stopped. These tests stub both the ANE
and GPU legs with identities, so a correct implementation must reproduce its
input row-for-row whichever leg handles the tail.
"""

import mlx.core as mx
import pytest

from omlx.patches import qwen35_ane_prefill as ane


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    """Clean environment, and drop the cached threshold so each test re-reads it."""
    monkeypatch.delenv("OMLX_QWEN35_ANE_TAIL_OVERLAP_MIN", raising=False)
    ane._reset_tail_overlap_min_cache()
    yield
    ane._reset_tail_overlap_min_cache()


def _rows(n, dim=8):
    """Distinguishable rows so any mis-slice shows up as a value mismatch."""
    return (mx.arange(n * dim, dtype=mx.float16) / 1000.0).reshape(1, n, dim)


class _StubMLP:
    def __init__(self, seq):
        self._omlx_ane_prefill_config = ane._AnePrefillConfig(seq, 0.6, 8, False)
        self.gate_proj = object()
        self.up_proj = object()
        self.down_proj = object()


@pytest.fixture
def identity_legs(monkeypatch):
    """ANE and GPU legs both become identities; record ANE call widths."""
    widths: list[int] = []

    def exact(module, x, target_verify=False):
        widths.append(int(x.shape[-2]))
        return x

    monkeypatch.setattr(ane, "_backend_exact", exact)
    monkeypatch.setattr(ane, "_tail_qmm_or_linear", lambda linear, x, variant: x)
    monkeypatch.setattr(ane, "swiglu", lambda gate, up: gate, raising=False)
    return widths


class TestThreshold:
    def test_default_is_off(self):
        assert ane._tail_overlap_min() == 0

    def test_reads_env(self, monkeypatch):
        ane._reset_tail_overlap_min_cache(); monkeypatch.setenv("OMLX_QWEN35_ANE_TAIL_OVERLAP_MIN", "720")
        assert ane._tail_overlap_min() == 720

    def test_is_resolved_once(self, monkeypatch):
        """The hot path must not re-read the environment on every call."""
        ane._reset_tail_overlap_min_cache(); monkeypatch.setenv("OMLX_QWEN35_ANE_TAIL_OVERLAP_MIN", "720")
        assert ane._tail_overlap_min() == 720
        monkeypatch.setenv("OMLX_QWEN35_ANE_TAIL_OVERLAP_MIN", "99")
        assert ane._tail_overlap_min() == 720, "value should be cached"
        ane._reset_tail_overlap_min_cache()
        assert ane._tail_overlap_min() == 99

    def test_garbage_is_off(self, monkeypatch):
        ane._reset_tail_overlap_min_cache(); monkeypatch.setenv("OMLX_QWEN35_ANE_TAIL_OVERLAP_MIN", "not-a-number")
        assert ane._tail_overlap_min() == 0

    def test_negative_is_clamped_off(self, monkeypatch):
        """A negative value must not enable the overlap for every tail."""
        ane._reset_tail_overlap_min_cache(); monkeypatch.setenv("OMLX_QWEN35_ANE_TAIL_OVERLAP_MIN", "-1")
        assert ane._tail_overlap_min() == 0


class TestTilePlan:
    @pytest.mark.parametrize(
        "rows,seq,expected",
        [(1190, 1024, (1, 166)), (2048, 1024, (2, 0)), (4090, 1024, (3, 1018)),
         (8079, 1024, (7, 911)), (1023, 1024, None), (4095, 2048, (1, 2047))],
    )
    def test_plan(self, rows, seq, expected):
        plan = ane._tiled_input_plan(_rows(rows), seq)
        assert plan == expected


class _StubGDN:
    def __init__(self, seq):
        self._omlx_ane_gdn_config = ane._AneGDNConfig(seq, 0.6, 8, False)
        self.in_proj_qkv = object()
        self.in_proj_z = object()
        self.in_proj_b = object()
        self.in_proj_a = object()


@pytest.fixture
def identity_gdn_legs(monkeypatch):
    """GDN ANE and GPU legs become identities; record ANE call widths."""
    widths: list[int] = []

    def exact(module, x, target_verify=False):
        widths.append(int(x.shape[-2]))
        return (x, x, x, x)

    monkeypatch.setattr(ane, "_gdn_backend_exact", exact)
    monkeypatch.setattr(ane, "_tail_qmm_or_linear", lambda linear, x, variant: x)
    return widths


class TestOverlapTailGDN:
    """GDN carries the same correctness argument as the MLP path, so it needs the
    same coverage: the projections are position-wise and the convolution and
    recurrent update happen outside this backend."""

    @pytest.mark.parametrize("rows,seq", [(1190, 1024), (4090, 1024), (8079, 1024)])
    def test_all_four_outputs_identical_with_and_without_overlap(
        self, monkeypatch, identity_gdn_legs, rows, seq
    ):
        gdn = _StubGDN(seq)
        x = _rows(rows)

        ane._reset_tail_overlap_min_cache(); monkeypatch.delenv("OMLX_QWEN35_ANE_TAIL_OVERLAP_MIN", raising=False)
        gpu_tail = ane._gdn_backend(gdn, x)

        ane._reset_tail_overlap_min_cache(); monkeypatch.setenv("OMLX_QWEN35_ANE_TAIL_OVERLAP_MIN", "1")
        overlapped = ane._gdn_backend(gdn, x)

        assert gpu_tail is not None and overlapped is not None
        assert len(gpu_tail) == len(overlapped) == 4
        for index in range(4):
            assert gpu_tail[index].shape == x.shape
            assert mx.array_equal(gpu_tail[index], x), f"gpu leg, output {index}"
            assert mx.array_equal(overlapped[index], x), f"overlap leg, output {index}"

    def test_overlap_runs_one_extra_full_tile(self, monkeypatch, identity_gdn_legs):
        gdn = _StubGDN(1024)
        ane._reset_tail_overlap_min_cache(); monkeypatch.setenv("OMLX_QWEN35_ANE_TAIL_OVERLAP_MIN", "720")
        ane._gdn_backend(gdn, _rows(4090))
        assert identity_gdn_legs == [1024, 1024, 1024, 1024], identity_gdn_legs

    def test_below_threshold_keeps_the_gpu_tail(self, monkeypatch, identity_gdn_legs):
        gdn = _StubGDN(1024)
        ane._reset_tail_overlap_min_cache(); monkeypatch.setenv("OMLX_QWEN35_ANE_TAIL_OVERLAP_MIN", "720")
        ane._gdn_backend(gdn, _rows(1190))   # 166-row tail, under the threshold
        assert identity_gdn_legs == [1024], identity_gdn_legs

    def test_declining_ane_leg_falls_back_without_losing_rows(self, monkeypatch):
        calls = {"n": 0}

        def exact(module, x, target_verify=False):
            calls["n"] += 1
            return None if calls["n"] > 3 else (x, x, x, x)

        monkeypatch.setattr(ane, "_gdn_backend_exact", exact)
        monkeypatch.setattr(ane, "_tail_qmm_or_linear", lambda linear, x, variant: x)
        ane._reset_tail_overlap_min_cache(); monkeypatch.setenv("OMLX_QWEN35_ANE_TAIL_OVERLAP_MIN", "1")
        out = ane._gdn_backend(_StubGDN(1024), _rows(4090))
        assert out is not None
        for index in range(4):
            assert mx.array_equal(out[index], _rows(4090)), index

    def test_target_verify_is_never_accelerated(self, monkeypatch, identity_gdn_legs):
        ane._reset_tail_overlap_min_cache(); monkeypatch.setenv("OMLX_QWEN35_ANE_TAIL_OVERLAP_MIN", "1")
        assert ane._gdn_backend(_StubGDN(1024), _rows(4090), True) is None
        assert identity_gdn_legs == []


class TestOverlapTailMLP:
    @pytest.mark.parametrize("rows,seq", [(1190, 1024), (4090, 1024), (8079, 1024)])
    def test_output_identical_with_and_without_overlap(
        self, monkeypatch, identity_legs, rows, seq
    ):
        """The overlap slice must reproduce the GPU-tail result exactly."""
        mlp = _StubMLP(seq)
        x = _rows(rows)

        ane._reset_tail_overlap_min_cache(); monkeypatch.delenv("OMLX_QWEN35_ANE_TAIL_OVERLAP_MIN", raising=False)
        gpu_tail = ane._backend(mlp, x)

        ane._reset_tail_overlap_min_cache(); monkeypatch.setenv("OMLX_QWEN35_ANE_TAIL_OVERLAP_MIN", "1")
        overlapped = ane._backend(mlp, x)

        assert gpu_tail is not None and overlapped is not None
        assert gpu_tail.shape == x.shape == overlapped.shape
        assert mx.array_equal(gpu_tail, x)
        assert mx.array_equal(overlapped, x)

    def test_overlap_runs_one_extra_full_tile(self, monkeypatch, identity_legs):
        mlp = _StubMLP(1024)
        x = _rows(4090)  # 3 full tiles + 1018 tail
        ane._reset_tail_overlap_min_cache(); monkeypatch.setenv("OMLX_QWEN35_ANE_TAIL_OVERLAP_MIN", "720")
        ane._backend(mlp, x)
        assert identity_legs == [1024, 1024, 1024, 1024], identity_legs

    def test_below_threshold_keeps_the_gpu_tail(self, monkeypatch, identity_legs):
        mlp = _StubMLP(1024)
        x = _rows(1190)  # 1 full tile + 166 tail, under the threshold
        ane._reset_tail_overlap_min_cache(); monkeypatch.setenv("OMLX_QWEN35_ANE_TAIL_OVERLAP_MIN", "720")
        ane._backend(mlp, x)
        assert identity_legs == [1024], identity_legs

    def test_threshold_boundary_is_inclusive(self, monkeypatch, identity_legs):
        """tail_rows == threshold takes the overlap, matching the >= in the code."""
        mlp = _StubMLP(1024)
        x = _rows(2048 + 166)          # 2 full tiles + a 166-row tail
        ane._reset_tail_overlap_min_cache(); monkeypatch.setenv("OMLX_QWEN35_ANE_TAIL_OVERLAP_MIN", "166")
        out = ane._backend(mlp, x)
        assert identity_legs == [1024, 1024, 1024], identity_legs
        assert mx.array_equal(out, x)

    def test_no_tail_is_untouched(self, monkeypatch, identity_legs):
        mlp = _StubMLP(1024)
        x = _rows(2048)  # exact multiple: no tail either way
        ane._reset_tail_overlap_min_cache(); monkeypatch.setenv("OMLX_QWEN35_ANE_TAIL_OVERLAP_MIN", "1")
        out = ane._backend(mlp, x)
        assert identity_legs == [1024, 1024]
        assert mx.array_equal(out, x)

    def test_overlap_falls_back_when_the_ane_leg_declines(self, monkeypatch):
        """A None from the exact backend must not lose the tail."""
        calls = {"n": 0}

        def exact(module, x, target_verify=False):
            calls["n"] += 1
            # full tiles succeed, the overlap tile declines
            return None if calls["n"] > 3 else x

        monkeypatch.setattr(ane, "_backend_exact", exact)
        monkeypatch.setattr(ane, "_tail_qmm_or_linear", lambda linear, x, variant: x)
        monkeypatch.setattr(ane, "swiglu", lambda gate, up: gate, raising=False)
        ane._reset_tail_overlap_min_cache(); monkeypatch.setenv("OMLX_QWEN35_ANE_TAIL_OVERLAP_MIN", "1")
        mlp = _StubMLP(1024)
        x = _rows(4090)
        out = ane._backend(mlp, x)
        assert out is not None and mx.array_equal(out, x)
