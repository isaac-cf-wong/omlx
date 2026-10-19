# SPDX-License-Identifier: Apache-2.0
"""Register ``mlx_lm.models.nemotron_h_puzzle`` (NVIDIA Nemotron 3.5 Bolt).

The model's routed and shared experts use the ``iq2_xxs`` format, which only
exists in this fork's ``omlx.custom_kernels.iq2_xxs``; see
``nemotron_h_puzzle_model`` for the architecture.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_QUALNAME = "mlx_lm.models.nemotron_h_puzzle"
_APPLIED = False


def apply_nemotron_h_puzzle_patch() -> bool:
    """Make mlx-lm resolve ``model_type: nemotron_h_puzzle``."""
    global _APPLIED
    if _APPLIED or _QUALNAME in sys.modules:
        return False

    file_path = Path(__file__).parent / "nemotron_h_puzzle_model.py"
    spec = importlib.util.spec_from_file_location(_QUALNAME, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {_QUALNAME} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "mlx_lm.models"
    sys.modules[_QUALNAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[_QUALNAME]
        raise
    importlib.import_module("mlx_lm.models").nemotron_h_puzzle = module
    _APPLIED = True
    logger.info("Registered %s", _QUALNAME)
    return True
