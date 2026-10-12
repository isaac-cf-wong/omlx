# SPDX-License-Identifier: Apache-2.0
"""Install the Qwen3.5-family prefill stack on a DFlash target.

``BatchedEngine`` and ``VLMBatchedEngine`` install the bounded SDPA256 route,
the FA-256 steel attention route, the native q4/q8 MLP and prefill-linear
tiles, and the private ANE/GPU prefill split before serving. Without this
module a DFlash target prefills with none of them.

dflash-mlx drives Qwen GDN targets through the mlx-lm module tree
(``mlx_lm.models.qwen3_5``), so only patches that reach that tree are
installed here. The mlx-vlm-only patches (GDN prework, chunked GDN, verify
SDPA split, ragged decode, the mlx-vlm prefill-linear sibling) are left to
``VLMBatchedEngine``, which installs them itself when DFlash falls back to it.

Install order matters. dflash-mlx replaces ``Attention.__call__`` and
``GatedDeltaNet.__call__`` on the target's classes and falls through to the
previous ``__call__`` for prefill. The class patches therefore install before
the target loads: dflash's hooks then wrap them, and
``restore_dflash_class_patches`` restores them rather than dropping them. The
ANE split compiles slices from the loaded model, so it runs after the load on
the MLX executor thread.

Each piece is gated by the same per-model setting as on the batched engines,
and the ANE arguments come from ``qwen35_ane_prefill_kwargs``, so a tuned
profile applies here exactly as it does there.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _enabled(model_settings: Any, name: str) -> bool:
    # Match the batched engines: these default on, and only an explicit False
    # disables them.
    return getattr(model_settings, name, True) is not False


def install_dflash_qwen35_class_patches(model_settings: Any) -> dict[str, bool]:
    """Install the class/module-level prefill patches, in the batched order.

    Call before ``load_target_bundle`` so dflash's hooks wrap these patches
    instead of being wrapped by them. Each patch is independent: one that
    fails to install is logged and the rest still apply.
    """
    installed: dict[str, bool] = {}

    def attempt(name: str, setting: str, install) -> None:
        if not _enabled(model_settings, setting):
            return
        try:
            installed[name] = bool(install())
        except Exception:
            installed[name] = False
            logger.debug(
                "DFlash Qwen prefill patch %s not applied", name, exc_info=True
            )

    def sdpa256() -> bool:
        from .sdpa256_attention import apply_sdpa256_attention_patch

        return apply_sdpa256_attention_patch()

    def fa256() -> bool:
        from .qwen35_fa256_attention import apply_qwen35_fa256_attention_patch

        return apply_qwen35_fa256_attention_patch()

    def q4_mlp() -> bool:
        from .qwen35_q4_mlp import apply_qwen35_q4_mlp_patch

        return apply_qwen35_q4_mlp_patch()

    def q4_lm_prefill_linear() -> bool:
        # The mlx-lm wrappers on Attention and GatedDeltaNet are the ones a
        # DFlash target executes; they also carry the ANE GDN backend.
        from .qwen35_q4_mlp import apply_qwen35_q4_lm_prefill_linear_patch

        return apply_qwen35_q4_lm_prefill_linear_patch()

    attempt("sdpa256", "sdpa256_prefill_enabled", sdpa256)
    attempt("fa256", "fa256_steel_prefill_enabled", fa256)
    attempt("q4_mlp", "qwen35_q4_mlp_prefill_enabled", q4_mlp)
    attempt(
        "q4_lm_prefill_linear", "qwen35_q4_mlp_prefill_enabled", q4_lm_prefill_linear
    )
    return installed


def enable_dflash_qwen35_ane(
    model: Any,
    model_settings: Any,
    model_type: str | None,
    *,
    prefill_step_size: int,
) -> int:
    """Compile the ANE prefill slices on a loaded DFlash target.

    Runs on the MLX executor thread after ``load_target_bundle``. Returns the
    number of MLP layers served on ANE; 0 when ANE prefill is off, unsupported
    for ``model_type``, or fails to compile, in which case prefill stays on
    the GPU.
    """
    from ..model_settings import ane_prefill_backend, validate_ane_prefill

    if not getattr(model_settings, "qwen35_ane_prefill_enabled", False):
        return 0
    if ane_prefill_backend(model_type) != "qwen":
        logger.info(
            "ANE prefill is enabled but has no DFlash backend for model type %s",
            model_type,
        )
        return 0
    try:
        from .qwen35_ane_prefill import (
            enable_qwen35_ane_prefill,
            qwen35_ane_prefill_kwargs,
        )

        validate_ane_prefill(model_settings.to_dict(), model_type)
        kwargs = qwen35_ane_prefill_kwargs(model_settings, model_type)
        count = enable_qwen35_ane_prefill(model, **kwargs)
    except Exception:
        logger.warning("Qwen ANE prefill not enabled for DFlash", exc_info=True)
        return 0

    gdn_count = int(getattr(model, "_omlx_ane_gdn_prefill_count", 0) or 0)
    sequence_length = kwargs["sequence_length"]
    if (count or gdn_count) and sequence_length > prefill_step_size:
        # dflash-mlx prefills in prefill_step_size chunks and never widens
        # them, so no chunk fills a wider compiled shape.
        logger.warning(
            "Qwen ANE prefill sequence_length=%d exceeds the DFlash prefill "
            "step (%d tokens); chunks narrower than the compiled shape need "
            "eligible tail padding to run on ANE.",
            sequence_length,
            prefill_step_size,
        )
    logger.info(
        "DFlash Qwen ANE prefill: %d MLP / %d GDN layers, tile=%d, step=%d",
        count,
        gdn_count,
        sequence_length,
        prefill_step_size,
    )
    return count


def release_dflash_qwen35_ane(model: Any) -> None:
    """Free the ANE banks a DFlash target holds. Call on the MLX executor."""
    if model is None:
        return
    try:
        from .qwen35_ane_prefill import release_qwen35_ane_prefill

        released, programs = release_qwen35_ane_prefill(model)
    except Exception:
        logger.warning("DFlash ANE prefill release failed", exc_info=True)
        return
    if released:
        logger.info(
            "Released %d DFlash ANE prefill module state(s) (%d program(s))",
            released,
            programs,
        )
