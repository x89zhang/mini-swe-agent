"""vLLM model convenience wrapper.

This uses text-based action parsing because local vLLM deployments often do not
expose reliable tool-calling behavior across model families.
"""

from __future__ import annotations

import os
from typing import Any

from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel


class VllmModel(LitellmTextbasedModel):
    """LiteLLM text-based model with vLLM-friendly defaults."""

    def __init__(self, **kwargs):
        default_kwargs: dict[str, Any] = {
            "api_base": os.getenv("MSWEA_VLLM_API_BASE", "http://localhost:8000/v1"),
            "api_key": os.getenv("MSWEA_VLLM_API_KEY", "EMPTY"),
        }
        model_kwargs = dict(default_kwargs)
        model_kwargs.update(kwargs.get("model_kwargs", {}))
        kwargs["model_kwargs"] = model_kwargs
        kwargs.setdefault("cost_tracking", "ignore_errors")
        super().__init__(**kwargs)
