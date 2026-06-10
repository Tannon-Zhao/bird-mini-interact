"""
Export native_fncall benchmark trajectories into demo_claw-style SFT JSONL.

Each results.jsonl entry is a serialized SampleStatus. For native_fncall runs it
carries native_base_messages ([system, user(query)]) plus interaction_history with
raw_assistant per turn. We replay that into a flat OpenAI-style message list:

    {"label": ..., "messages": [system, user, assistant, user(tool_response), ...],
     "meta": {...}}

The shape matches mini_interact/demo_claw.jsonl, so the output can be concatenated
into the next SFT round directly.

Usage:
    python -m batch_run_bird_interact.export_native_trajectory \
        --results_jsonl outputs/batch_runs/.../results.jsonl \
        --output outputs/sft_roundNNN.jsonl \
        --label bird_interact_native_sft \
        --filter_reward gt=0
"""

import argparse
import json

import jsonlines

from batch_run_bird_interact.native_messages_builder import (
    _assistant_content_from_turn,
    _tool_response_content,
)


def _reward_passes(meta_reward, expr) -> bool:
    """expr like 'gt=0', 'ge=1', 'eq=1'; None expr means accept all."""
    if not expr:
        return True
    try:
        op, val = expr.split("=", 1)
        val = float(val)
    except ValueError:
        return True
    r = float(meta_reward or 0.0)
    return {
        "gt": r > val, "ge": r >= val, "lt": r < val,
        "le": r <= val, "eq": r == val, "ne": r != val,
    }.get(op, True)


def status_to_sft_sample(status: dict, label: str) -> dict:
    """Replay one serialized SampleStatus into a flat SFT message list."""
    base = list(status.get("native_base_messages") or [])
    messages = list(base)
    for turn_log in status.get("interaction_history", []):
        messages.append({"role": "assistant", "content": _assistant_content_from_turn(turn_log)})
        messages.append({"role": "user", "content": _tool_response_content(turn_log)})
    # Drop the trailing tool_response (no assistant reply follows it).
    if messages and messages[-1]["role"] == "user":
        messages.pop()
    return {
        "label": label,
        "messages": messages,
        "meta": {
            "idx": status.get("idx"),
            "reward": status.get("last_reward"),
            "phase1_completed": status.get("phase1_completed"),
            "phase2_completed": status.get("phase2_completed"),
            "task_finished": status.get("task_finished"),
            "selected_database": (status.get("original_data") or {}).get("selected_database"),
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Export native_fncall trajectories to SFT JSONL.")
    ap.add_argument("--results_jsonl", required=True, help="Path to a native_fncall results.jsonl")
    ap.add_argument("--output", required=True, help="Output SFT JSONL path")
    ap.add_argument("--label", default="bird_interact_native_sft", help="label field for each sample")
    ap.add_argument("--filter_reward", default=None,
                    help="Keep samples whose last_reward satisfies OP=VAL, e.g. gt=0, ge=1, eq=1")
    ap.add_argument("--require_native", action="store_true",
                    help="Skip statuses without native_base_messages (legacy react_text rows)")
    args = ap.parse_args()

    kept, skipped = 0, 0
    with jsonlines.open(args.results_jsonl, mode="r") as reader, \
         jsonlines.open(args.output, mode="w") as writer:
        for status in reader:
            if args.require_native and not status.get("native_base_messages"):
                skipped += 1
                continue
            if not _reward_passes(status.get("last_reward"), args.filter_reward):
                skipped += 1
                continue
            sample = status_to_sft_sample(status, args.label)
            if len(sample["messages"]) < 3:  # need at least system+user+one assistant
                skipped += 1
                continue
            writer.write(sample)
            kept += 1

    print(f"Wrote {kept} SFT samples to {args.output} (skipped {skipped}).")


if __name__ == "__main__":
    main()
