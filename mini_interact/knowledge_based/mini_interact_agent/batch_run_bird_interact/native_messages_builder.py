"""
Builds demo-style (native function-calling) message arrays for the BIRD-Interact
agent when running with --agent_chat_mode=native_fncall.

Data flow (per sample, per turn):
  build_native_initial_messages(status, ...)  -> [system, user(query)]  (once)
  get_native_messages_for_turn(status, ...)   -> base + expanded history
      assistant turns : raw_assistant if available, else reconstructed from
                        (thought, action_string)
      user turns      : <tool_response> wrapper around the observation

The parser (prompt_utils.parse_agent_response) already understands the
<think>+<tool_call> output, so the model's raw output round-trips back into the
benchmark's (thought, interaction_object, action_string) triple unchanged.
"""

import hashlib
import json
import re
from typing import Dict, List, Optional

from experiments.utils.prompts_native import TemplateNativeFnCallBirdInteract

# Mirror prompt_utils.SETTING_MAP / LANGUAGE_MAP (kept local to avoid a cycle).
SETTING_MAP = {
    "bird_interact_sql": "PostgreSQL Database",
    "mini_interact": "SQLite Database",
}
LANGUAGE_MAP = {
    "bird_interact_sql": "PostgreSQL",
    "mini_interact": "SQLite",
}

TRUNC_MARKER = "\n[... earlier output truncated ...]\n"


def _action_string_to_tool_name(action_str: str) -> str:
    """`execute("...")` -> `execute`; tolerant of leading/trailing whitespace."""
    if not action_str:
        return "unknown"
    head = action_str.strip().split("(", 1)[0].strip()
    return head if head else "unknown"


def _gen_tool_call_id(turn: int, action_str: str) -> str:
    """Deterministic demo-style id. Determinism keeps resume reproducible."""
    h = hashlib.sha1(f"{turn}|{action_str}".encode("utf-8")).hexdigest()[:16]
    return f"chatcmpltool{h}"


def _action_string_to_tool_call_json(thought: str, action_str: str) -> str:
    """Inverse of prompt_utils._format_action_call: action string -> tool_call JSON.

    Only used as a fallback when raw_assistant is missing (e.g. resuming a run
    that predates raw_assistant logging). Arguments are JSON-encoded, so quotes
    inside SQL/questions are escaped correctly.
    """
    action_str = (action_str or "").strip()
    name = _action_string_to_tool_name(action_str)

    def _inner(s: str) -> str:
        # text between the first '(' and the matching last ')'
        lp = s.find("(")
        rp = s.rfind(")")
        if lp == -1 or rp == -1 or rp < lp:
            return ""
        return s[lp + 1 : rp].strip()

    arguments: Dict[str, object] = {}
    if name in ("execute", "submit"):
        arguments = {"sql": _inner(action_str).strip().strip("'\"")}
    elif name == "ask":
        arguments = {"question": _inner(action_str).strip().strip("'\"")}
    elif name == "get_column_meaning":
        parts = [p.strip().strip("'\"") for p in _inner(action_str).split(",", 1)]
        arguments = {
            "table_name": parts[0] if len(parts) > 0 else "",
            "column_name": parts[1] if len(parts) > 1 else "",
        }
    elif name == "get_knowledge_definition":
        arguments = {"knowledge_name": _inner(action_str).strip().strip("'\"")}
    # all other tools take no arguments

    return json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)


def _assistant_content_from_turn(turn_log: dict) -> str:
    """Prefer the model's original output; reconstruct only as a fallback."""
    raw = turn_log.get("raw_assistant")
    if raw:
        return raw
    thought = turn_log.get("thought", "") or ""
    tc_json = _action_string_to_tool_call_json(thought, turn_log.get("action", ""))
    return f"<think>\n{thought}\n</think>\n<tool_call>\n{tc_json}\n</tool_call>"


def _tool_response_content(turn_log: dict, observation_override: Optional[str] = None) -> str:
    """Wrap an observation in the demo-style <tool_response> envelope."""
    action = turn_log.get("action", "") or ""
    tool_name = _action_string_to_tool_name(action)
    tool_call_id = turn_log.get("tool_call_id") or _gen_tool_call_id(
        turn_log.get("turn", 0), action
    )
    observation = observation_override if observation_override is not None else turn_log.get("observation", "")
    return (
        "<tool_response>\n"
        f"<name>{tool_name}</name>\n"
        f"<id>{tool_call_id}</id>\n"
        "<result>\n"
        f"{observation}\n"
        "</result>\n"
        "</tool_response>"
    )


def build_native_initial_messages(status, budget_info: dict, env_type: str) -> List[Dict[str, str]]:
    """Construct [system, user(query)] and store on status.native_base_messages."""
    tmpl = TemplateNativeFnCallBirdInteract(LANGUAGE_MAP[env_type], SETTING_MAP[env_type])
    query = status.original_data["amb_user_query"]
    messages = [
        {"role": "system", "content": tmpl.get_native_init_msg()},
        {"role": "user", "content": tmpl.get_query_msg(query, budget_info)},
    ]
    status.native_base_messages = messages
    return messages


def get_native_messages_for_turn(status, max_prompt_chars: Optional[int] = None) -> List[Dict[str, str]]:
    """Expand base messages + interaction history into a chat-mode message array.

    Truncation strategy mirrors SampleStatus.get_truncated_interaction_prompt:
    keep system/query/all assistant turns intact; progressively replace the
    oldest <tool_response> result bodies with a truncation marker until the
    total character count fits within max_prompt_chars.
    """
    base = list(status.native_base_messages or [])

    history = status.interaction_history or []
    # Each history turn => one assistant message + one tool_response user message.
    assistant_msgs = [_assistant_content_from_turn(t) for t in history]
    observations = [t.get("observation", "") for t in history]

    def assemble(obs_list: List[str]) -> List[Dict[str, str]]:
        msgs = list(base)
        for t, a_content, obs in zip(history, assistant_msgs, obs_list):
            msgs.append({"role": "assistant", "content": a_content})
            msgs.append({"role": "user", "content": _tool_response_content(t, observation_override=obs)})
        return msgs

    messages = assemble(observations)

    if max_prompt_chars is None:
        return messages

    def total_chars(msgs: List[Dict[str, str]]) -> int:
        return sum(len(m["content"]) for m in msgs)

    if total_chars(messages) <= max_prompt_chars:
        return messages

    # Truncate oldest observations first.
    obs_list = list(observations)
    for i in range(len(obs_list)):
        obs_list[i] = TRUNC_MARKER.strip()
        candidate = assemble(obs_list)
        if total_chars(candidate) <= max_prompt_chars:
            return candidate

    return assemble(obs_list)


if __name__ == "__main__":
    # Minimal self-test with a fake status object.
    class _FakeStatus:
        def __init__(self):
            self.original_data = {"amb_user_query": "Find several calibrated alien signals."}
            self.native_base_messages = None
            self.interaction_history = []

    s = _FakeStatus()
    build_native_initial_messages(s, {"total_budget": 22.0}, "mini_interact")
    s.interaction_history = [
        {
            "turn": 1, "thought": "Check the schema first.",
            "interaction_object": "Environment", "action": "get_schema()",
            "observation": "CREATE TABLE Signals (...);",
            "raw_assistant": '<think>\nCheck the schema first.\n</think>\n<tool_call>\n{"name": "get_schema", "arguments": {}}\n</tool_call>',
            "tool_call_id": "chatcmpltoolabc123",
        },
        {
            "turn": 2, "thought": "Submit final SQL.",
            "interaction_object": "User",
            "action": 'submit("SELECT s.SignalID FROM Signals s WHERE s.SignalStrength > 10")',
            "observation": "Your SQL is correct!",
            "raw_assistant": None,  # exercise the reconstruction fallback
        },
    ]
    msgs = get_native_messages_for_turn(s, max_prompt_chars=None)
    print(json.dumps(msgs, indent=2, ensure_ascii=False))
    print("\n--- truncated (max 600 chars) ---\n")
    print(json.dumps(get_native_messages_for_turn(s, max_prompt_chars=600), indent=2, ensure_ascii=False))
