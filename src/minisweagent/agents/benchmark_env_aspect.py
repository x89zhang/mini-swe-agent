"""Benchmark helpers for replacing bash input/output at the environment layer."""

from __future__ import annotations

import re
from typing import Any

try:
    import aspectlib
except Exception:  # pragma: no cover - optional dependency
    aspectlib = None  # type: ignore[assignment]


def _ensure_aspectlib() -> None:
    if aspectlib is None:
        raise ImportError(
            "aspectlib is required for benchmark_env_aspect; install it with `pip install aspectlib`."
        )


def _apply_rules(text: str, rules: list[dict[str, Any]]) -> str:
    for rule in rules:
        pattern = str(rule.get("pattern", ""))
        replace = str(rule.get("replace", ""))
        if rule.get("regex", True):
            text = re.sub(pattern, replace, text)
        else:
            text = text.replace(pattern, replace)
    return text


def install_benchmark_env_bash_io_aspect(
    *,
    command_replacements: list[dict[str, Any]] | None = None,
    output_replacements: list[dict[str, Any]] | None = None,
    env_class: type | None = None,
):
    """Install an aspect that rewrites command input/output around Environment.execute.

    Rule format: ``{"pattern": "...", "replace": "...", "regex": true}``
    """
    _ensure_aspectlib()
    if env_class is None:
        from minisweagent.environments.local import LocalEnvironment

        env_class = LocalEnvironment

    cmd_rules = command_replacements or []
    out_rules = output_replacements or []

    @aspectlib.Aspect
    def _around_execute(*args, **kwargs):
        if len(args) >= 2:
            env = args[0]
            action = args[1]
            rest = args[2:]
            call_kwargs = dict(kwargs)
        else:
            env = args[0]
            action = kwargs.get("action", {})
            rest = ()
            call_kwargs = dict(kwargs)

        updated_action = dict(action) if isinstance(action, dict) else {"command": str(action)}
        updated_action["command"] = _apply_rules(str(updated_action.get("command", "")), cmd_rules)

        result = yield aspectlib.Proceed(env, updated_action, *rest, **call_kwargs)
        if isinstance(result, dict) and "output" in result:
            result = dict(result)
            result["output"] = _apply_rules(str(result.get("output", "")), out_rules)
        yield aspectlib.Return(result)

    return aspectlib.weave(env_class.execute, _around_execute)
