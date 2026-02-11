"""Benchmark helpers for replacing bash input/output with aspectlib.

This module targets the agent layer by weaving ``DefaultAgent.execute_actions``.
It can rewrite action commands before execution and rewrite command outputs
before they are formatted back into observation messages.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from typing import Any

try:
    import aspectlib
except Exception:  # pragma: no cover - optional dependency
    aspectlib = None  # type: ignore[assignment]

Rule = dict[str, Any]
ActionTransform = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
OutputTransform = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def _ensure_aspectlib() -> None:
    if aspectlib is None:
        raise ImportError(
            "aspectlib is required for benchmark_aspect; install it with `pip install aspectlib`."
        )


def _apply_rules(text: str, rules: list[Rule]) -> str:
    for rule in rules:
        pattern = str(rule.get("pattern", ""))
        replace = str(rule.get("replace", ""))
        if rule.get("regex", True):
            text = re.sub(pattern, replace, text)
        else:
            text = text.replace(pattern, replace)
    return text


def _build_action_transform(rules: list[Rule] | None) -> ActionTransform | None:
    if not rules:
        return None

    def _transform(action: dict[str, Any], _meta: dict[str, Any]) -> dict[str, Any]:
        transformed = dict(action)
        transformed["command"] = _apply_rules(str(transformed.get("command", "")), rules)
        return transformed

    return _transform


def _build_output_transform(rules: list[Rule] | None) -> OutputTransform | None:
    if not rules:
        return None

    def _transform(output: dict[str, Any], _meta: dict[str, Any]) -> dict[str, Any]:
        transformed = dict(output)
        transformed["output"] = _apply_rules(str(transformed.get("output", "")), rules)
        return transformed

    return _transform


def install_benchmark_bash_io_aspect(
    *,
    action_replacements: list[Rule] | None = None,
    output_replacements: list[Rule] | None = None,
    transform_action: ActionTransform | None = None,
    transform_output: OutputTransform | None = None,
    agent_class: type | None = None,
):
    """Install an aspect that rewrites bash commands and outputs.

    You can pass either:
    - ``action_replacements`` / ``output_replacements`` (regex replacement rules), or
    - ``transform_action`` / ``transform_output`` (custom callbacks),
      or both (custom callbacks run after rule-based replacements).

    Replacement rule format:
    ``{"pattern": "...", "replace": "...", "regex": true}``
    """
    _ensure_aspectlib()
    if agent_class is None:
        from minisweagent.agents.default import DefaultAgent

        agent_class = DefaultAgent

    rule_action_transform = _build_action_transform(action_replacements)
    rule_output_transform = _build_output_transform(output_replacements)

    @aspectlib.Aspect
    def _around_execute_actions(*args, **kwargs):
        if len(args) >= 2:
            agent = args[0]
            message = args[1]
        else:
            agent = args[0]
            message = kwargs["message"]

        msg = copy.deepcopy(message)
        actions = msg.get("extra", {}).get("actions", [])
        rewritten_actions = []
        for action in actions:
            updated = dict(action)
            meta = {"agent": agent, "message": msg}
            if rule_action_transform is not None:
                updated = rule_action_transform(updated, meta)
            if transform_action is not None:
                updated = transform_action(updated, meta)
            rewritten_actions.append(updated)

        msg.setdefault("extra", {})["actions"] = rewritten_actions
        outputs = [agent.env.execute(action) for action in rewritten_actions]
        rewritten_outputs = []
        for output in outputs:
            updated = dict(output)
            meta = {"agent": agent, "message": msg}
            if rule_output_transform is not None:
                updated = rule_output_transform(updated, meta)
            if transform_output is not None:
                updated = transform_output(updated, meta)
            rewritten_outputs.append(updated)

        observation_messages = agent.model.format_observation_messages(
            msg, rewritten_outputs, agent.get_template_vars()
        )
        result = agent.add_messages(*observation_messages)
        yield aspectlib.Return(result)

    return aspectlib.weave(agent_class.execute_actions, _around_execute_actions)
