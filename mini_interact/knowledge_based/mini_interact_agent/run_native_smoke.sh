#!/bin/bash
# run_native_smoke.sh
# 冒烟测试:用 GLM-5.1(siliconflow)以 native function-calling 模式跑 mini_interact benchmark,
# 看 student 协议(<think>+<tool_call> 多轮 messages)下的生成轨迹。
#
# 用法:
#   bash run_native_smoke.sh                # 默认跑 3 条样本
#   LIMIT=10 bash run_native_smoke.sh       # 跑 10 条
#   AGENT_MODEL=qwen3-30b-a3b-student-native AGENT_BASE_URL=http://localhost:8000/v1 \
#     AGENT_API_KEY=dummy bash run_native_smoke.sh   # 换成自训 student(本地 vllm)
#
# 注意:在 comp 上跑(数据集 + sqlite DB 都在那)。这是全新 run,不要加 --resume。

set -e
cd "$(dirname "$0")"

# ---- 可被环境变量覆盖的配置 ----
AGENT_MODEL="${AGENT_MODEL:-Pro/zai-org/GLM-5.1}"
AGENT_API_KEY="${AGENT_API_KEY:-sk-zpnqmypcqsoecmgibosdhkzxgqwxfimaxysnroeyehasgbuk}"
AGENT_BASE_URL="${AGENT_BASE_URL:-https://api.siliconflow.cn/v1}"
USER_MODEL="${USER_MODEL:-Pro/deepseek-ai/DeepSeek-V3.1-Terminus}"
DATA_PATH="${DATA_PATH:-/home/tyzhao/workspace/project/BIRD-Interact/mini_interact/knowledge_based/data/mini-interact/bird_interact_data_with_gt.jsonl}"
BUDGETS="${BUDGETS:-6}"
MAX_TURNS="${MAX_TURNS:-16}"
LIMIT="${LIMIT:-3}"

AGENT_MODEL_SAFE=$(echo "$AGENT_MODEL" | sed 's/[^a-zA-Z0-9_-]/-/g')
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$(pwd)/outputs/batch_new/native_smoke_${AGENT_MODEL_SAFE}}"
mkdir -p "$BASE_OUTPUT_DIR"

echo "====================================================="
echo " Native function-calling smoke test"
echo "   Agent model : $AGENT_MODEL"
echo "   Base URL    : $AGENT_BASE_URL"
echo "   User model  : $USER_MODEL"
echo "   Samples     : $LIMIT (budget=$BUDGETS, max_turns=$MAX_TURNS)"
echo "   Output dir  : $BASE_OUTPUT_DIR"
echo "====================================================="

AGENT_API_KEY="$AGENT_API_KEY" AGENT_BASE_URL="$AGENT_BASE_URL" \
bash run_batch_mini_interact_experiments.sh \
    --data_path="$DATA_PATH" \
    --agent_models="$AGENT_MODEL" \
    --agent_chat_mode=native_fncall \
    --user_model="$USER_MODEL" \
    --budgets="$BUDGETS" \
    --max_turns="$MAX_TURNS" \
    --base_output_dir="$BASE_OUTPUT_DIR" \
    --mini_interact

RESULTS="${BASE_OUTPUT_DIR}/${AGENT_MODEL_SAFE}_patience_${BUDGETS}/results.jsonl"
echo ""
echo "====================================================="
echo " Done. Results: $RESULTS"
echo "====================================================="

# ---- 打印每条轨迹的工具调用序列 ----
if [ -f "$RESULTS" ]; then
    echo ""
    echo "--- Trajectory summary ---"
    python - "$RESULTS" <<'PY'
import json, sys
path = sys.argv[1]
for line in open(path, encoding="utf-8"):
    s = json.loads(line)
    print(f"=== idx {s['idx']} | finished={s['task_finished']} | reward={s['last_reward']} "
          f"| phase1={s.get('phase1_completed')}")
    for t in s.get("interaction_history", []):
        act = (t.get("action") or "").replace("\n", " ")[:90]
        invalid = "  <-- INVALID FORMAT" if "Invalid action format" in (t.get("observation") or "") else ""
        print(f"   turn{t['turn']} [{t['interaction_object']}] {act}{invalid}")
PY
    echo ""
    echo "To export demo_claw-style SFT data from this run:"
    echo "  python -m batch_run_bird_interact.export_native_trajectory \\"
    echo "      --results_jsonl $RESULTS \\"
    echo "      --output ${BASE_OUTPUT_DIR}/native_trajectories_sft.jsonl \\"
    echo "      --label bird_interact_native_sft"
fi
