"""
Convert round014 ReAct-XML SFT data (with distilled <think>) into the native
Qwen3 Hermes function-calling protocol used by --agent_chat_mode=native_fncall.

Two input envelope shapes (see round_014_data):
  - standard multi-turn (label react_sft):
      [system, user(toy demo + query + budget), assistant(<think>+ReAct XML), user(Observation:..), ...]
    -> [system(native init), user(native demo + query + budget),
        assistant(<think>+<tool_call>{json}), user(<tool_response>), ...]  (drop trailing lone user)
    Every assistant turn carries loss (full trajectory).

  - one-step recovery (label react_sft_recovery):
      [user(full pre-rendered system+demo+query+history text), assistant(correction)]
    -> [system(native init),
        user(native demo + query + budget + native-rendered history text),
        assistant(native correction)]
    Loss only on the corrective assistant turn (faithful to round014's design):
    the history is rendered as native TEXT inside the user message, not as
    separate assistant/tool role turns.

The system prompt / demo / query / tool_response wrappers come from the SAME
TemplateNativeFnCallBirdInteract used by the live benchmark, guaranteeing
training distribution == eval protocol.
"""

import argparse
import json
import re
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils.prompts_native import TemplateNativeFnCallBirdInteract
from batch_run_bird_interact.native_messages_builder import (
    _action_string_to_tool_call_json,
    _action_string_to_tool_name,
    _gen_tool_call_id,
)
from batch_run_bird_interact.prompt_utils import parse_agent_response_ex
from batch_run_bird_interact.action_format_utils import is_valid_parsed_action

# SQLite (mini_interact) env, matching the benchmark's LANGUAGE_MAP/SETTING_MAP.
TMPL = TemplateNativeFnCallBirdInteract("SQLite", "SQLite Database")
KNOWN_TOOLS = {
    "execute", "get_schema", "get_all_column_meanings", "get_column_meaning",
    "get_all_external_knowledge_names", "get_knowledge_definition",
    "get_all_knowledge_definitions", "ask", "submit",
}

THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)
ACTION_RE = re.compile(r'<action>(.*?)</action>', re.DOTALL)
QUERY_RE = re.compile(r"User's Question:\s*(.*?)\n\s*:", re.DOTALL)
BUDGET_RE = re.compile(r'total action budget of\s*([\d.]+)\s*units')
# one rendered ReAct history step inside a pre-rendered prompt blob.
# Anchor the observation on its trailing "[SYSTEM NOTE: ...]" budget note rather
# than the next <thought>, so observations that themselves contain literal
# <thought>/<action> tags (e.g. INVALID_ACTION_FORMAT messages) don't corrupt
# step segmentation.
STEP_RE = re.compile(
    r'<thought>(?P<thought>.*?)</thought>\s*'
    r'<interaction_object>(?P<io>.*?)</interaction_object>\s*'
    r'<action>(?P<action>.*?)</action>\s*'
    r'Observation:\s*(?P<obs>.*?\[SYSTEM NOTE:[^\]]*\])',
    re.DOTALL,
)


class ConvertError(Exception):
    pass


def clean_action(raw: str) -> str:
    """Normalize an extracted action string: strip nested/duplicated <action> tags.

    Some source turns have doubled tags like `<action><action>execute(...)</action>`,
    which a non-greedy <action>(.*?)</action> captures as `<action>execute(...)`.
    Take the content after the LAST <action> open tag and before its </action>.
    """
    raw = raw.strip()
    if '<action>' in raw:
        raw = raw.split('<action>')[-1]
    if '</action>' in raw:
        raw = raw.split('</action>')[0]
    return raw.strip()


def extract_query_and_budget(text: str):
    """Pull the real user question + total budget out of a prompt blob / user msg."""
    # query: prefer the one after TASK START; fall back to the last match.
    task_part = text.split("# -----TASK START-----", 1)
    search_in = task_part[1] if len(task_part) > 1 else text
    m = QUERY_RE.search(search_in)
    if not m:
        raise ConvertError("no User's Question found")
    query = m.group(1).strip()
    bm = BUDGET_RE.search(text)
    budget = float(bm.group(1)) if bm else 18.0
    return query, budget


def action_to_tool_json(action_str: str):
    """action string -> (tool_call_json_str, name). Raise if name not a known tool."""
    tj = _action_string_to_tool_call_json("", action_str)
    obj = json.loads(tj)
    name = obj.get("name", "")
    if name not in KNOWN_TOOLS:
        raise ConvertError(f"unknown tool from action: {action_str[:60]!r}")
    return tj, name


def make_assistant_content(think: str, action_str: str):
    """Build <think>..</think>\\n<tool_call>{json}</tool_call>; validate round-trip."""
    tj, name = action_to_tool_json(action_str)
    content = f"<think>\n{think.strip()}\n</think>\n<tool_call>\n{tj}\n</tool_call>"
    # round-trip: must parse back to a valid action with the same tool name
    _, _, parsed_act, _ = parse_agent_response_ex(content)
    if not is_valid_parsed_action(parsed_act):
        raise ConvertError(f"round-trip invalid: {parsed_act[:60]!r}")
    if _action_string_to_tool_name(parsed_act) != name:
        raise ConvertError(f"round-trip name mismatch: {parsed_act[:60]!r} != {name}")
    return content, name


def make_tool_response(obs_body: str, prev_action: str, turn: int):
    """Wrap an observation (Observation: prefix stripped) in <tool_response>."""
    name = _action_string_to_tool_name(prev_action)
    tid = _gen_tool_call_id(turn, prev_action)
    body = re.sub(r'^\s*Observation:\s*', '', obs_body).rstrip()
    return (
        "<tool_response>\n"
        f"<name>{name}</name>\n"
        f"<id>{tid}</id>\n"
        "<result>\n"
        f"{body}\n"
        "</result>\n"
        "</tool_response>"
    )


def first_user_content(query: str, budget: float, extra_history: str = ""):
    base = TMPL.get_demos_native() + "\n\n" + TMPL.get_query_msg(query, {"total_budget": budget})
    if extra_history:
        base += "\n\n" + extra_history
    return base


def render_history_text(steps, start_turn=1):
    """Render (thought, action, obs) steps as native text for embedding in a user msg."""
    blocks = []
    for i, (thought, action_str, obs) in enumerate(steps):
        a_content, _ = make_assistant_content(thought, clean_action(action_str))
        tr = make_tool_response(obs, clean_action(action_str), start_turn + i)
        blocks.append(a_content + "\n\n" + tr)
    return "\n\n".join(blocks)


def convert_standard(sample):
    msgs = sample["messages"]
    if not msgs or msgs[0]["role"] != "system":
        raise ConvertError("standard: no system message")
    # message[1] is user with demo + query + budget
    if len(msgs) < 3 or msgs[1]["role"] != "user":
        raise ConvertError("standard: unexpected head")
    query, budget = extract_query_and_budget(msgs[1]["content"])

    out = [
        {"role": "system", "content": TMPL.get_native_init_msg()},
        {"role": "user", "content": first_user_content(query, budget)},
    ]
    turn = 1
    i = 2
    while i < len(msgs):
        m = msgs[i]
        if m["role"] == "assistant":
            tm = THINK_RE.search(m["content"])
            am = ACTION_RE.search(m["content"])
            if not am:
                raise ConvertError("standard: assistant without <action>")
            think = tm.group(1) if tm else ""
            action_str = clean_action(am.group(1))
            content, _ = make_assistant_content(think, action_str)
            out.append({"role": "assistant", "content": content})
            # following user/observation (if any) becomes a tool_response
            if i + 1 < len(msgs) and msgs[i + 1]["role"] == "user":
                tr = make_tool_response(msgs[i + 1]["content"], action_str, turn)
                out.append({"role": "user", "content": tr})
                i += 2
            else:
                i += 1
            turn += 1
        else:
            # stray user without preceding assistant -> skip
            i += 1
    # drop trailing lone user (final observation with no assistant after)
    if out and out[-1]["role"] == "user":
        out.pop()
    if out[-1]["role"] != "assistant":
        raise ConvertError("standard: no assistant turns produced")
    return out


def convert_recovery(sample):
    msgs = sample["messages"]
    if len(msgs) != 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
        raise ConvertError("recovery: unexpected shape")
    blob = msgs[0]["content"]
    query, budget = extract_query_and_budget(blob)

    # history region = everything from the first <thought> after TASK START budget note
    task_part = blob.split("# -----TASK START-----", 1)
    region = task_part[1] if len(task_part) > 1 else blob
    first_t = region.find("<thought>")
    steps = []
    if first_t != -1:
        for mm in STEP_RE.finditer(region[first_t:]):
            steps.append((mm.group("thought"), mm.group("action").strip(), mm.group("obs")))
    history_text = render_history_text(steps) if steps else ""

    # corrective step (the single loss-bearing assistant turn)
    corr = msgs[1]["content"]
    tm = THINK_RE.search(corr)
    am = ACTION_RE.search(corr)
    if not am:
        raise ConvertError("recovery: correction without <action>")
    corr_content, _ = make_assistant_content(tm.group(1) if tm else "", clean_action(am.group(1)))

    out = [
        {"role": "system", "content": TMPL.get_native_init_msg()},
        {"role": "user", "content": first_user_content(query, budget, history_text)},
        {"role": "assistant", "content": corr_content},
    ]
    return out


def convert_sample(sample):
    label = sample.get("label", "")
    if sample["messages"] and sample["messages"][0]["role"] == "system":
        new_msgs = convert_standard(sample)
    else:
        new_msgs = convert_recovery(sample)
    return {
        "label": label + "_native" if label and not label.endswith("_native") else label or "native",
        "task_id": sample.get("task_id"),
        "db_id": sample.get("db_id"),
        "status": sample.get("status"),
        "source": sample.get("source"),
        "protocol": "native_fncall",
        "messages": new_msgs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--include-recovery", action="store_true",
                    help="also convert react_sft_recovery (one-step) samples")
    args = ap.parse_args()

    kept = dropped = 0
    drop_reasons = Counter()
    tool_dist = Counter()
    shape = Counter()
    with open(args.inp, encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for ln in fin:
            ln = ln.strip()
            if not ln:
                continue
            sample = json.loads(ln)
            is_recovery = not (sample["messages"] and sample["messages"][0]["role"] == "system")
            if is_recovery and not args.include_recovery:
                dropped += 1
                drop_reasons["recovery-excluded"] += 1
                continue
            try:
                conv = convert_sample(sample)
            except ConvertError as e:
                dropped += 1
                drop_reasons[str(e)[:40]] += 1
                continue
            except Exception as e:  # noqa
                dropped += 1
                drop_reasons["EXC:" + type(e).__name__] += 1
                continue
            shape["recovery" if is_recovery else "standard"] += 1
            for m in conv["messages"]:
                if m["role"] == "assistant":
                    _, _, a, _ = parse_agent_response_ex(m["content"])
                    tool_dist[_action_string_to_tool_name(a)] += 1
            fout.write(json.dumps(conv, ensure_ascii=False) + "\n")
            kept += 1

    print(f"kept={kept} dropped={dropped}")
    print("by shape:", dict(shape))
    print("drop reasons:", dict(drop_reasons))
    print("tool distribution (assistant turns w/ loss):", dict(tool_dist))


if __name__ == "__main__":
    main()
