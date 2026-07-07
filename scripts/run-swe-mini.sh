#!/usr/bin/env bash
set -euo pipefail

# Run pi-bench SWE Mini tasks using the upstream pi-bench submodule plus
# WebUI-owned customizations: generated models.json, Lemonade judge model
# resolution, and result ownership repair.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PI_BENCH_DIR="${PI_BENCH_DIR:-$REPO_ROOT/third_party/pi-bench}"
PI_BENCH_RUN_DIR="${PI_BENCH_RUN_DIR:-$PI_BENCH_DIR}"
REGISTRY="${SWE_BENCH_IMAGE_REGISTRY:-ghcr.io/epoch-research/swe-bench.eval.x86_64}"

usage() {
	cat >&2 <<'USAGE'
Usage:
  scripts/run-swe-mini.sh <task-file-or-dir> [pi-bench args...]

Environment:
  PI_BENCH_DIR=/path/to/pi-bench-submodule
  PI_BENCH_RUN_DIR=/path/to/writable/pi-bench-runtime  # optional
  PI_BENCH_MODELS_JSON=/path/to/generated/models.json
USAGE
}

if [ $# -lt 1 ]; then
	usage
	exit 1
fi

TARGET="$1"
shift

PASS_COUNT=1
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
	case "$1" in
	--pass)
		PASS_COUNT="$2"
		shift 2
		;;
	*)
		EXTRA_ARGS+=("$1")
		shift
		;;
	esac
done

# shellcheck source=scripts/docker-ownership.sh
source "$SCRIPT_DIR/docker-ownership.sh"

if [ ! -d "$PI_BENCH_DIR" ]; then
	echo "[ERROR] PI_BENCH_DIR does not exist: $PI_BENCH_DIR" >&2
	exit 1
fi

if [ "$PI_BENCH_RUN_DIR" != "$PI_BENCH_DIR" ]; then
	mkdir -p "$PI_BENCH_RUN_DIR"
	# Keep the runtime copy simple and dependency-free. This intentionally avoids
	# deleting files so cached node_modules and previous results can be reused.
	cp -a "$PI_BENCH_DIR/." "$PI_BENCH_RUN_DIR/"
fi

patch_pi_bench_for_local_judge_models() {
	local index_ts="$PI_BENCH_RUN_DIR/src/index.ts"
	[ -f "$index_ts" ] || return 0
	python3 - "$index_ts" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
marker = "LMEVAL_WEBUI_LOCAL_JUDGE_MODEL_RESOLUTION"
if marker in text:
    raise SystemExit(0)
old_judge_parse = '''  let judgeModelReq;
  if (values["judge-model"]) {
    const parts = values["judge-model"].split("/");
    judgeModelReq = parts.length > 1 ? getModel(parts[0] as any, parts[1]) : undefined;
    if (!judgeModelReq && !values["print-output-dir"]) console.warn(`[WARN] Could not resolve judge model ${values["judge-model"]}. Using default.`);
  }
'''
new_judge_parse = '''  let judgeModelReq;
  if (values["judge-model"]) {
    const parts = values["judge-model"].split("/");
    if (parts.length > 1) {
      const judgeProvider = parts[0] as any;
      const judgeId = parts.slice(1).join("/");
      judgeModelReq = getModel(judgeProvider, judgeId) || { provider: judgeProvider, id: judgeId };
    }
  }
'''
if old_judge_parse not in text:
    raise SystemExit(f"Could not patch {path}: judge parser not found")
text = text.replace(old_judge_parse, new_judge_parse, 1)
needle = """    }

    let resolvedAgentModel;
"""
insert = """    }

    // LMEVAL_WEBUI_LOCAL_JUDGE_MODEL_RESOLUTION
    if (judgeModelReq && !judgeModelReq.api) {
      const resolvedJudgeModel = modelRegistry.find(judgeModelReq.provider, judgeModelReq.id);
      if (!resolvedJudgeModel) {
        throw new Error(`Could not find judge model ${judgeModelReq.provider}/${judgeModelReq.id} in registry`);
      }
      judgeModelReq = resolvedJudgeModel;
      console.log(`[INFO] Judge resolved to model: ${judgeModelReq.provider}/${judgeModelReq.id}`);
    }

    let resolvedAgentModel;
"""
if needle not in text:
    raise SystemExit(f"Could not patch {path}: insertion point not found")
path.write_text(text.replace(needle, insert, 1), encoding="utf-8")
PY
}

patch_pi_bench_for_local_judge_models

MODEL_DOCKER_ARGS=()
if [ -n "${PI_BENCH_MODELS_JSON:-}" ]; then
	if [ ! -f "$PI_BENCH_MODELS_JSON" ]; then
		echo "[ERROR] PI_BENCH_MODELS_JSON does not exist: $PI_BENCH_MODELS_JSON" >&2
		exit 1
	fi
	MODEL_DOCKER_ARGS=(-v "$PI_BENCH_MODELS_JSON:/pi-bench/models.json:z,ro")
fi

ENV_ARGS=()
if [ -f "$REPO_ROOT/.env" ]; then
	ENV_ARGS=(--env-file "$REPO_ROOT/.env")
elif [ -f "$PI_BENCH_RUN_DIR/.env" ]; then
	ENV_ARGS=(--env-file "$PI_BENCH_RUN_DIR/.env")
fi

mkdir -p "$PI_BENCH_RUN_DIR"
docker volume create pi-bench-bun-cache >/dev/null 2>&1 || true

resolve_target() {
	local raw="$1"
	if [ -e "$raw" ]; then
		realpath "$raw"
	elif [ -e "$PI_BENCH_RUN_DIR/$raw" ]; then
		realpath "$PI_BENCH_RUN_DIR/$raw"
	elif [ -e "$PI_BENCH_DIR/$raw" ]; then
		realpath "$PI_BENCH_DIR/$raw"
	else
		printf '%s\n' "$raw"
	fi
}

TARGET_ABS="$(resolve_target "$TARGET")"
TASK_FILES=()
if [ -d "$TARGET_ABS" ]; then
	for f in "$TARGET_ABS"/*.json; do
		[ -f "$f" ] && TASK_FILES+=("$f")
	done
else
	TASK_FILES+=("$TARGET_ABS")
fi

if [ ${#TASK_FILES[@]} -eq 0 ]; then
	echo "[ERROR] No task JSON files found in $TARGET" >&2
	exit 1
fi

if [ -n "${SWE_MINI_OUTPUT_PATH:-}" ]; then
	RESULTS_DIR="$(
		python3 - "$SWE_MINI_OUTPUT_PATH" "$PI_BENCH_RUN_DIR" <<'PY'
import os, sys
print(os.path.relpath(sys.argv[1], sys.argv[2]))
PY
	)"
else
	set +e
	RESULTS_DIR="$(cd "$PI_BENCH_RUN_DIR" && bun run src/index.ts --print-output-dir "$TARGET" "${EXTRA_ARGS[@]}" 2>/dev/null)"
	set -e
fi

TOTAL=${#TASK_FILES[@]}
COUNT=0
PASSED=0
FAILED=0

printf '========================================================\n'
printf '[INFO] SWE Mini Runner — %s tasks queued\n' "$TOTAL"
printf '[INFO] pi-bench source: %s\n' "$PI_BENCH_DIR"
printf '[INFO] pi-bench runtime: %s\n' "$PI_BENCH_RUN_DIR"
[ -n "${RESULTS_DIR:-}" ] && printf '[INFO] Results directory: %s\n' "$RESULTS_DIR"
printf '========================================================\n'

for task_file in "${TASK_FILES[@]}"; do
	COUNT=$((COUNT + 1))
	TASK_ID="$(
		python3 - "$task_file" <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf-8') as handle:
    print(json.load(handle)['id'])
PY
	)"
	IMAGE="$REGISTRY.$TASK_ID:latest"

	if [ -n "${RESULTS_DIR:-}" ] && [ -f "$PI_BENCH_RUN_DIR/$RESULTS_DIR/results-${TASK_ID}.json" ]; then
		restore_docker_result_ownership "$RESULTS_DIR" "$IMAGE"
		printf '\n========================================================\n'
		printf '[%s/%s] Task: %s\n' "$COUNT" "$TOTAL" "$TASK_ID"
		printf '[INFO] Skipping %s, result already exists.\n' "$TASK_ID"
		printf '========================================================\n'
		EXISTING_SCORE="$(
			python3 - "$PI_BENCH_RUN_DIR/$RESULTS_DIR/results-${TASK_ID}.json" <<'PY' || true
import json, sys
with open(sys.argv[1], encoding='utf-8') as handle:
    print(json.load(handle).get('judgeScore', 0))
PY
		)"
		if [ "$EXISTING_SCORE" = "1" ]; then
			PASSED=$((PASSED + 1))
		else
			FAILED=$((FAILED + 1))
		fi
		continue
	fi

	printf '\n========================================================\n'
	printf '[%s/%s] Task: %s\n' "$COUNT" "$TOTAL" "$TASK_ID"
	printf '         Image: %s\n' "$IMAGE"
	printf '========================================================\n'

	REL_TASK_FILE="$(
		python3 - "$task_file" "$PI_BENCH_RUN_DIR" <<'PY'
import os, sys
print(os.path.relpath(sys.argv[1], sys.argv[2]))
PY
	)"

	for ATTEMPT in $(seq 1 "$PASS_COUNT"); do
		if [ "$PASS_COUNT" -gt 1 ]; then
			echo "[INFO] Starting attempt $ATTEMPT of $PASS_COUNT for $TASK_ID"
		fi

		LOGFILE="$(mktemp /tmp/lm-eval-swe-mini-log.XXXXXX)"
		set +e
		docker run --init -i --rm --network host \
			"${ENV_ARGS[@]}" \
			-v "$PI_BENCH_RUN_DIR:/pi-bench:z" \
			"${MODEL_DOCKER_ARGS[@]}" \
			-v "pi-bench-bun-cache:/root/.bun" \
			"$IMAGE" \
			bash -c "
        set -e
        if [ ! -f /root/.bun/bin/bun ]; then
          echo '[SETUP] Installing bun...'
          apt-get update -qq && apt-get install -y -qq unzip >/dev/null 2>&1
          curl -fsSL https://bun.sh/install | bash >/dev/null 2>&1
          echo '[SETUP] bun installed.'
        fi
        export PATH=/root/.bun/bin:\$PATH
        which unzip >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq unzip >/dev/null 2>&1; }
        cd /pi-bench && bun install --frozen-lockfile 2>/dev/null || bun install 2>/dev/null
        source /opt/miniconda3/etc/profile.d/conda.sh
        conda activate testbed
        bun run src/index.ts '$REL_TASK_FILE' ${EXTRA_ARGS[*]@Q}
      " 2>&1 | tee "$LOGFILE"
		EXIT_CODE=${PIPESTATUS[0]}
		set -e

		if [ -z "${RESULTS_DIR:-}" ]; then
			RESULTS_DIR="$(grep -m1 'Saving results to directory:' "$LOGFILE" | sed 's/.*Saving results to directory: //' | tr -d '\r' || true)"
		fi
		rm -f "$LOGFILE"
		restore_docker_result_ownership "$RESULTS_DIR" "$IMAGE"

		if [ "$EXIT_CODE" -eq 2 ]; then
			echo "[FATAL] Inference backend is unreachable or crashed. Aborting entire benchmark run."
			exit 2
		fi
		if [ "$EXIT_CODE" -ne 0 ]; then
			echo "[FATAL] SWE Mini container exited with status $EXIT_CODE."
			exit "$EXIT_CODE"
		fi
		if [ -z "${RESULTS_DIR:-}" ]; then
			echo "[FATAL] No results directory produced for task $TASK_ID."
			exit 1
		fi

		RESULT_FILE="$PI_BENCH_RUN_DIR/$RESULTS_DIR/results-${TASK_ID}.json"
		if [ ! -f "$RESULT_FILE" ]; then
			echo "[FATAL] No result file produced for task $TASK_ID: $RESULT_FILE"
			exit 1
		fi
		mv "$RESULT_FILE" "$PI_BENCH_RUN_DIR/$RESULTS_DIR/results-${TASK_ID}-attempt${ATTEMPT}.json"
		mv "$PI_BENCH_RUN_DIR/$RESULTS_DIR/transcript-${TASK_ID}.json" "$PI_BENCH_RUN_DIR/$RESULTS_DIR/transcript-${TASK_ID}-attempt${ATTEMPT}.json" 2>/dev/null || true
		JUDGE_SCORE="$(
			python3 - "$PI_BENCH_RUN_DIR/$RESULTS_DIR/results-${TASK_ID}-attempt${ATTEMPT}.json" <<'PY' || true
import json, sys
with open(sys.argv[1], encoding='utf-8') as handle:
    print(json.load(handle).get('judgeScore', 0))
PY
		)"
		if [ "$JUDGE_SCORE" = "1" ]; then
			break
		fi
	done

	python3 - "$PI_BENCH_RUN_DIR/$RESULTS_DIR" "$TASK_ID" "$PASS_COUNT" <<'PY'
import json, os, shutil, sys
results_dir, task_id, pass_count_s = sys.argv[1], sys.argv[2], sys.argv[3]
pass_count = int(pass_count_s)
attempts = []
best_attempt = None
succeeded_at = None
for attempt in range(1, pass_count + 1):
    path = os.path.join(results_dir, f"results-{task_id}-attempt{attempt}.json")
    if not os.path.exists(path):
        continue
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    attempts.append(data)
    best_attempt = attempt
    if data.get("judgeScore") == 1:
        succeeded_at = attempt
        break
if attempts:
    final_data = attempts[-1].copy()
    final_data["attempts"] = attempts
    final_data["succeededAtAttempt"] = succeeded_at
    with open(os.path.join(results_dir, f"results-{task_id}.json"), "w", encoding="utf-8") as handle:
        json.dump(final_data, handle, indent=2)
    best_transcript = os.path.join(results_dir, f"transcript-{task_id}-attempt{best_attempt}.json")
    final_transcript = os.path.join(results_dir, f"transcript-{task_id}.json")
    if os.path.exists(best_transcript):
        shutil.copy2(best_transcript, final_transcript)
PY

	FINAL_SCORE="$(
		python3 - "$PI_BENCH_RUN_DIR/$RESULTS_DIR/results-${TASK_ID}.json" <<'PY' || true
import json, sys
with open(sys.argv[1], encoding='utf-8') as handle:
    print(json.load(handle).get('judgeScore', 0))
PY
	)"
	if [ "$FINAL_SCORE" = "1" ]; then
		PASSED=$((PASSED + 1))
	else
		FAILED=$((FAILED + 1))
		echo "[WARN] Task $TASK_ID failed"
	fi
done

printf '\n========================================================\n'
printf '[INFO] SWE Mini Runner Complete!\n'
printf '[INFO] Tasks: %s | Succeeded: %s | Failed: %s\n' "$TOTAL" "$PASSED" "$FAILED"
printf '========================================================\n'

if [ -n "${RESULTS_DIR:-}" ] && [ -d "$PI_BENCH_RUN_DIR/$RESULTS_DIR" ]; then
	echo "[INFO] Generating aggregate summary from $RESULTS_DIR ..."
	python3 - "$PI_BENCH_RUN_DIR/$RESULTS_DIR" <<'PY'
import glob, json, os, sys
results_dir = sys.argv[1]
result_files = [
    path for path in sorted(glob.glob(os.path.join(results_dir, "results-*.json")))
    if "-attempt" not in os.path.basename(path)
]
if not result_files:
    print("[WARN] No result files found, skipping summary generation.")
    sys.exit(0)
results = []
passed = 0
total_duration = 0
for path in result_files:
    with open(path, encoding="utf-8") as handle:
        result = json.load(handle)
    results.append(result)
    if result.get("judgeScore") == 1:
        passed += 1
    total_duration += result.get("durationMs", 0)
summary = {
    "totalTasks": len(results),
    "passedTasks": passed,
    "passRate": passed / len(results) if results else 0,
    "totalDurationMs": total_duration,
    "averageDurationMs": total_duration / len(results) if results else 0,
    "results": results,
}
summary_path = os.path.join(results_dir, "summary.json")
with open(summary_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)
print(f"[INFO] Aggregate summary: {passed}/{len(results)} passed ({summary['passRate']*100:.1f}%)")
print(f"[INFO] Summary saved to {summary_path}")
PY
fi
