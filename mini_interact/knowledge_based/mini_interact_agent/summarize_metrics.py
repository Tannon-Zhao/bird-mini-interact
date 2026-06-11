#!/usr/bin/env python3
"""Scan ./outputs for results_metrics.json, dump four summary files at project root.

Writes:
  metrics_summary.json    -- all runs, full metrics + path metadata
  metrics_by_model.json   -- groupby model -> config (b?m?) -> list of runs (total_reward + key fields)
  metrics_table.csv       -- row=model, col=b?m?, cell=avg_reward (most-recent run for that pair)
  metrics_table.md        -- same matrix as csv, GFM table

All files include updated_at ISO timestamp (json) / a metadata line (csv/md).
Run from project root, or pass --root explicitly.
"""
import argparse, csv, io, json, os, re, sys, time
from collections import defaultdict
from datetime import datetime, timezone

KNOWN_MODEL_MAP = {
    "Qwen-Qwen3-5-35B-A3B": "Qwen/Qwen3.5-35B-A3B",
    "Qwen-Qwen3-5-27B": "Qwen/Qwen3.5-27B",
    "Qwen-Qwen3-5-9B": "Qwen/Qwen3.5-9B",
    "Qwen-Qwen3-5-4B": "Qwen/Qwen3.5-4B",
    "Qwen-Qwen3-5-122B-A10B": "Qwen/Qwen3.5-122B-A10B",
    "Qwen-Qwen3-5-397B-A17B": "Qwen/Qwen3.5-397B-A17B",
    "Qwen-Qwen3-30B-A3B-Instruct-2507": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen-Qwen3-30B-A3B-Thinking-2507": "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "Pro-zai-org-GLM-5-1": "Pro/zai-org/GLM-5.1",
    "Pro-deepseek-ai-DeepSeek-V3-1-Terminus": "Pro/deepseek-ai/DeepSeek-V3.1-Terminus",
    "deepseek-ai-DeepSeek-V4-Flash": "deepseek-ai/DeepSeek-V4-Flash",
    "deepseek-ai-DeepSeek-V4-Flash-think": "deepseek-ai/DeepSeek-V4-Flash#think",
    "deepseek-ai-DeepSeek-V4-Flash-max": "deepseek-ai/DeepSeek-V4-Flash#max",
}

CONFIG_RE = re.compile(r"\bb(\d+)m(\d+)\b")
PATIENCE_RE = re.compile(r"^(.*?)_patience_(\d+)$")
SAMPLE_ID_RE = re.compile(r"Sample (\d+):")

ALLOWED_CONFIGS = {"b4m12", "b6m18", "b6m20"}


def classify_model(model):
    """Return (deploy, mode) for a model."""
    if model.startswith(("Qwen/", "Pro/", "deepseek-ai/")):
        deploy = "API"
    else:
        deploy = "local"
    lower = model.lower()
    if any(k in lower for k in ("think", "#max", "-r013think", "-r014think")):
        mode = "think"
    else:
        mode = "nonthink"
    return deploy, mode


def count_log_errors(run_dir_abs):
    """Parse experiment.log for unique samples with API errors or malformed actions.
    Returns (api_error_samples, malformed_action_samples) as int counts.
    Each sample ID is counted at most once per error type."""
    log_path = os.path.join(run_dir_abs, "experiment.log")
    api_err_ids = set()
    malformed_ids = set()
    if not os.path.isfile(log_path):
        return 0, 0
    try:
        with open(log_path, errors="replace") as f:
            for line in f:
                if "Agent API failure" in line:
                    m = SAMPLE_ID_RE.search(line)
                    if m:
                        api_err_ids.add(int(m.group(1)))
                elif "Rejected malformed action" in line:
                    m = SAMPLE_ID_RE.search(line)
                    if m:
                        malformed_ids.add(int(m.group(1)))
    except Exception:
        pass
    return len(api_err_ids), len(malformed_ids)


def parse_run(metrics_path, scan_root):
    """Derive (run_dir, config, model, patience) from the metrics file location."""
    run_dir_abs = os.path.dirname(metrics_path)
    rel = os.path.relpath(run_dir_abs, scan_root)
    parts = rel.split(os.sep)

    leaf = parts[-1]
    m = PATIENCE_RE.match(leaf)
    if m:
        model_dir = m.group(1)
        patience = int(m.group(2))
    else:
        model_dir = leaf
        patience = None

    model = KNOWN_MODEL_MAP.get(model_dir, model_dir)

    config = None
    max_turns = None
    for p in parts:
        cm = CONFIG_RE.fullmatch(p)
        if cm:
            config = "b%sm%s" % (cm.group(1), cm.group(2))
            max_turns = int(cm.group(2))
            break
    if config is None:
        config = "unknown"

    api_err, malformed = count_log_errors(run_dir_abs)

    return {
        "run_dir": rel.replace(os.sep, "/"),
        "config": config,
        "max_turns": max_turns,
        "model": model,
        "model_dir_name": model_dir,
        "patience": patience,
        "api_error_samples": api_err,
        "malformed_action_samples": malformed,
    }


def assess_completeness(metrics, max_turns):
    if max_turns is None:
        return False, "config has no parseable max_turns"
    td = metrics.get("turn_distribution") or {}
    try:
        keys = [int(k) for k in td.keys()]
    except (TypeError, ValueError):
        return False, "turn_distribution malformed"
    if not keys:
        return False, "turn_distribution empty"
    mx = max(keys)
    if mx < max_turns:
        return False, "max_turn_in_dist=%d < expected %d (run truncated)" % (mx, max_turns)
    if metrics.get("total_reward", 0) == 0 and metrics.get("avg_turns", 99) < 2:
        return False, "total_reward=0 and avg_turns<2 (run did not actually execute)"
    return True, "ok"


def load_metrics(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"_load_error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.getcwd())
    ap.add_argument("--scan", default=None)
    ap.add_argument("--summary-out", default=None)
    ap.add_argument("--groupby-out", default=None)
    ap.add_argument("--csv-out", default=None)
    ap.add_argument("--md-out", default=None)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    scan = os.path.abspath(args.scan) if args.scan else os.path.join(root, "outputs")
    summary_path = args.summary_out or os.path.join(root, "metrics_summary.json")
    groupby_path = args.groupby_out or os.path.join(root, "metrics_by_model.json")
    csv_path = args.csv_out or os.path.join(root, "metrics_table.csv")
    md_path = args.md_out or os.path.join(root, "metrics_table.md")

    if not os.path.isdir(scan):
        print("[err] scan dir not found: %s" % scan, file=sys.stderr)
        sys.exit(2)

    runs = []
    skipped_configs = 0
    skipped_incomplete = []
    for dirpath, _, files in os.walk(scan):
        if "results_metrics.json" in files:
            mp = os.path.join(dirpath, "results_metrics.json")
            meta = parse_run(mp, scan)
            if meta["config"] not in ALLOWED_CONFIGS:
                skipped_configs += 1
                continue
            meta["metrics_file"] = os.path.relpath(mp, scan).replace(os.sep, "/")
            meta["metrics_file_mtime"] = datetime.fromtimestamp(
                os.path.getmtime(mp), tz=timezone.utc).isoformat()
            meta["metrics"] = load_metrics(mp)
            is_complete, reason = assess_completeness(meta["metrics"], meta["max_turns"])
            meta["is_complete"] = is_complete
            meta["incomplete_reason"] = None if is_complete else reason
            if not is_complete:
                skipped_incomplete.append((meta["run_dir"], reason))
            runs.append(meta)

    runs.sort(key=lambda r: (r["model"], r["config"], r["patience"] or 0, r["metrics_file_mtime"]))
    updated_at = datetime.now(timezone.utc).isoformat()

    summary = {
        "updated_at": updated_at,
        "scan_root": scan,
        "run_count": len(runs),
        "runs": runs,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    grouped = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if not r["is_complete"]:
            continue
        m = r["metrics"]
        entry = {
            "run_dir": r["run_dir"],
            "patience": r["patience"],
            "metrics_file_mtime": r["metrics_file_mtime"],
            "total_reward": m.get("total_reward"),
            "phase1_completion_rate": m.get("phase1_completion_rate"),
            "phase1_completed": m.get("phase1_completed"),
            "completion_rate": m.get("completion_rate"),
            "completed_samples": m.get("completed_samples"),
            "total_samples": m.get("total_samples"),
            "avg_turns": m.get("avg_turns"),
            "avg_reward": m.get("avg_reward"),
            "api_error_samples": r.get("api_error_samples", 0),
            "malformed_action_samples": r.get("malformed_action_samples", 0),
        }
        grouped[r["model"]][r["config"]].append(entry)

    grouped_out = {}
    for model, cfgs in grouped.items():
        cfg_out = {}
        for cfg, items in cfgs.items():
            items.sort(key=lambda x: x["metrics_file_mtime"])
            rewards = [it["total_reward"] for it in items if isinstance(it["total_reward"], (int, float))]
            agg = {
                "n_runs": len(items),
                "total_reward_max": max(rewards) if rewards else None,
                "total_reward_min": min(rewards) if rewards else None,
                "total_reward_mean": (sum(rewards) / len(rewards)) if rewards else None,
                "total_reward_last": items[-1]["total_reward"] if items else None,
                "api_error_samples_total": sum(it.get("api_error_samples", 0) for it in items),
                "malformed_action_samples_total": sum(it.get("malformed_action_samples", 0) for it in items),
            }
            cfg_out[cfg] = {"aggregate": agg, "runs": items}
        grouped_out[model] = cfg_out

    groupby = {
        "updated_at": updated_at,
        "scan_root": scan,
        "models": grouped_out,
    }
    with open(groupby_path, "w") as f:
        json.dump(groupby, f, indent=2, ensure_ascii=False)

    # --- avg_reward matrix (csv + md) ---
    def _cfg_sort_key(c):
        m = CONFIG_RE.fullmatch(c)
        if m:
            return (0, int(m.group(1)), int(m.group(2)))
        return (1, 0, 0)

    all_configs = sorted({cfg for cfgs in grouped_out.values() for cfg in cfgs}, key=_cfg_sort_key)
    pinned = ["b4m12", "b6m18", "b6m20"]
    all_configs = [c for c in pinned if c in ALLOWED_CONFIGS]
    models_sorted = sorted(m for m in grouped_out if any(c in grouped_out[m] for c in all_configs))

    matrix = {}
    for model in models_sorted:
        row = {}
        for cfg in all_configs:
            payload = grouped_out.get(model, {}).get(cfg)
            if not payload:
                row[cfg] = None
                continue
            row[cfg] = payload["aggregate"].get("total_reward_mean")
        matrix[model] = row

    def _fmt(v):
        if v is None:
            return ""
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return str(v)
        return str(int(fv)) if fv == int(fv) else "%.2f" % fv

    # CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["# updated_at: %s" % updated_at])
        w.writerow(["# cell: total_reward (mean over n_runs) for (model, config); empty = no run"])
        w.writerow(["model", "deploy", "mode"] + all_configs + ["n_runs_by_config"])
        for model in models_sorted:
            cells = [_fmt(matrix[model][c]) for c in all_configs]
            n_str = "/".join(
                str(grouped_out[model].get(c, {}).get("aggregate", {}).get("n_runs", 0))
                for c in all_configs
            )
            deploy, mode = classify_model(model)
            w.writerow([model, deploy, mode] + cells + [n_str])

    # Markdown
    buf = io.StringIO()
    buf.write("# BIRD mini_interact -- total_reward matrix\n\n")
    buf.write("_updated_at: %s_\n\n" % updated_at)
    buf.write("_cell: total_reward, averaged across n_runs for that (model, config); empty = no run_\n\n")
    header_cols = ["model", "deploy", "mode"] + all_configs + ["n_runs"]
    buf.write("| " + " | ".join(header_cols) + " |\n")
    buf.write("|" + "|".join(["---"] * len(header_cols)) + "|\n")
    for model in models_sorted:
        cells = [_fmt(matrix[model][c]) or "—" for c in all_configs]
        n_str = "/".join(
            str(grouped_out[model].get(c, {}).get("aggregate", {}).get("n_runs", 0))
            for c in all_configs
        )
        deploy, mode = classify_model(model)
        row_vals = [model, deploy, mode] + cells + [n_str]
        buf.write("| " + " | ".join(row_vals) + " |\n")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())

    print("OK  runs=%d skipped_non_mainstream=%d skipped_incomplete=%d models=%d configs=%d" % (
        len(runs), skipped_configs, len(skipped_incomplete), len(models_sorted), len(all_configs)))
    if skipped_incomplete:
        print("  incomplete runs excluded from aggregates:")
        for rd, why in skipped_incomplete:
            print("    - %s  (%s)" % (rd, why))
    print("    summary=%s" % summary_path)
    print("    groupby=%s" % groupby_path)
    print("    csv=%s" % csv_path)
    print("    md=%s" % md_path)


if __name__ == "__main__":
    main()
