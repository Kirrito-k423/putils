#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python"
SNIFFER_SCRIPT="tools/python_stack_sniffer.py"
TRACE_OUTPUT="stack_trace.json"
SAMPLE_INTERVAL="0.1"
AUTOSAVE_INTERVAL="0"
ALL_THREADS=0
DRY_RUN=0
DETACH=0
ACTION="run"
PID_FILE=""
DETACH_LOG_FILE=""
EXTRA_SNIFFER_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  python_stack_sniffer_with_job.sh [options] -- <main job command>
  python_stack_sniffer_with_job.sh --status [--pid-file <file>]
  python_stack_sniffer_with_job.sh --stop [--pid-file <file>]

Options:
  --python <path>                 Python executable (default: python)
  --sniffer-script <path>         Sniffer script path (default: tools/python_stack_sniffer.py)
  --output <file>                 Trace output path (default: stack_trace.json)
  --interval <seconds>            Sniffer sample interval (default: 0.1)
  --autosave-interval <seconds>   Sniffer autosave interval (default: 0)
  --all-threads                   Pass --all-threads to sniffer
  --sniffer-arg <arg>             Extra sniffer argument (repeatable)
  --detach                        Async mode: return immediately after launch
  --status                        Show detached main/sniffer process status from pid file
  --stop                          Stop detached main/sniffer processes from pid file
  --pid-file <file>               PID metadata file (default: <output>.pids)
  --detach-log-file <file>        Detached wrapper log file (default: <output>.wrapper.log)
  --dry-run                       Print commands only, do not execute
  -h, --help                      Show this help

Examples:
  bash tools/python_stack_sniffer_with_job.sh \
    --output /tmp/trace.json --interval 0.2 --all-threads \
    -- python train.py

  bash tools/python_stack_sniffer_with_job.sh \
    --detach --output /tmp/trace.json \
    --sniffer-arg --gpu-usage --sniffer-arg --cpu-mem-usage \
    -- python train.py --epochs 10
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --sniffer-script)
      SNIFFER_SCRIPT="$2"
      shift 2
      ;;
    --output)
      TRACE_OUTPUT="$2"
      shift 2
      ;;
    --interval)
      SAMPLE_INTERVAL="$2"
      shift 2
      ;;
    --autosave-interval)
      AUTOSAVE_INTERVAL="$2"
      shift 2
      ;;
    --all-threads)
      ALL_THREADS=1
      shift
      ;;
    --sniffer-arg)
      EXTRA_SNIFFER_ARGS+=("$2")
      shift 2
      ;;
    --detach)
      DETACH=1
      shift
      ;;
    --status)
      ACTION="status"
      shift
      ;;
    --stop)
      ACTION="stop"
      shift
      ;;
    --pid-file)
      PID_FILE="$2"
      shift 2
      ;;
    --detach-log-file)
      DETACH_LOG_FILE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

MAIN_CMD=()
if [[ "$ACTION" == "run" ]]; then
  if [[ $# -eq 0 ]]; then
    echo "[ERROR] Missing main job command after --" >&2
    usage
    exit 2
  fi
  MAIN_CMD=("$@")
elif [[ $# -gt 0 ]]; then
  echo "[ERROR] Unexpected positional arguments for action: $ACTION" >&2
  usage
  exit 2
fi

if [[ -z "$PID_FILE" ]]; then
  PID_FILE="${TRACE_OUTPUT}.pids"
fi
if [[ -z "$DETACH_LOG_FILE" ]]; then
  DETACH_LOG_FILE="${TRACE_OUTPUT}.wrapper.log"
fi

TRACE_DIR="$(dirname "$TRACE_OUTPUT")"
PID_DIR="$(dirname "$PID_FILE")"
DETACH_LOG_DIR="$(dirname "$DETACH_LOG_FILE")"
mkdir -p "$TRACE_DIR" "$PID_DIR" "$DETACH_LOG_DIR"

pid_value() {
  local key="$1"
  if [[ ! -f "$PID_FILE" ]]; then
    return 1
  fi

  while IFS='=' read -r k v; do
    if [[ "$k" == "$key" ]]; then
      printf '%s\n' "$v"
      return 0
    fi
  done < "$PID_FILE"

  return 1
}

pid_running() {
  local pid="$1"
  if [[ -z "$pid" ]]; then
    return 1
  fi
  kill -0 "$pid" 2>/dev/null
}

run_status() {
  local main_pid=""
  local sniffer_pid=""
  main_pid="$(pid_value main_pid || true)"
  sniffer_pid="$(pid_value sniffer_pid || true)"

  echo "[INFO] PID file: $PID_FILE"
  if [[ -z "$main_pid" && -z "$sniffer_pid" ]]; then
    echo "[INFO] No tracked pids found"
    return 1
  fi

  if pid_running "$main_pid"; then
    echo "[INFO] main_pid=$main_pid running"
  else
    echo "[INFO] main_pid=$main_pid not-running"
  fi

  if pid_running "$sniffer_pid"; then
    echo "[INFO] sniffer_pid=$sniffer_pid running"
  else
    echo "[INFO] sniffer_pid=$sniffer_pid not-running"
  fi
}

run_stop() {
  local main_pid=""
  local sniffer_pid=""
  main_pid="$(pid_value main_pid || true)"
  sniffer_pid="$(pid_value sniffer_pid || true)"

  if [[ -z "$main_pid" && -z "$sniffer_pid" ]]; then
    echo "[INFO] No tracked pids found in $PID_FILE"
    return 0
  fi

  if pid_running "$main_pid"; then
    echo "[INFO] Stopping main job pid=$main_pid"
    kill -TERM "$main_pid" 2>/dev/null || true
  else
    echo "[INFO] Main job pid=$main_pid already stopped"
  fi

  if pid_running "$sniffer_pid"; then
    echo "[INFO] Stopping sniffer pid=$sniffer_pid"
    kill -INT "$sniffer_pid" 2>/dev/null || true
  else
    echo "[INFO] Sniffer pid=$sniffer_pid already stopped"
  fi
}

run_sync() {
  local sniffer_pid=""
  cleanup_sync() {
    if [[ -n "$sniffer_pid" ]] && kill -0 "$sniffer_pid" 2>/dev/null; then
      echo "[INFO] Stopping sniffer pid=$sniffer_pid"
      kill -INT "$sniffer_pid" 2>/dev/null || true
      wait "$sniffer_pid" 2>/dev/null || true
    fi
  }

  trap cleanup_sync EXIT INT TERM

  echo "[INFO] Starting main job: ${MAIN_CMD[*]}"
  "${MAIN_CMD[@]}" &
  local main_pid=$!
  echo "[INFO] Main job pid=$main_pid"

  local -a sniffer_cmd=(
    "$PYTHON_BIN" "$SNIFFER_SCRIPT"
    -p "$main_pid"
    -i "$SAMPLE_INTERVAL"
    -o "$TRACE_OUTPUT"
    --autosave-interval "$AUTOSAVE_INTERVAL"
  )
  if [[ "$ALL_THREADS" -eq 1 ]]; then
    sniffer_cmd+=(--all-threads)
  fi
  if [[ ${#EXTRA_SNIFFER_ARGS[@]} -gt 0 ]]; then
    sniffer_cmd+=("${EXTRA_SNIFFER_ARGS[@]}")
  fi

  echo "[INFO] Starting sniffer: ${sniffer_cmd[*]}"
  "${sniffer_cmd[@]}" &
  sniffer_pid=$!
  echo "[INFO] Sniffer pid=$sniffer_pid"

  {
    echo "main_pid=$main_pid"
    echo "sniffer_pid=$sniffer_pid"
  } > "$PID_FILE"

  set +e
  wait "$main_pid"
  local main_exit_code=$?
  set -e

  echo "[INFO] Main job finished with exit code: $main_exit_code"
  cleanup_sync
  sniffer_pid=""

  exit "$main_exit_code"
}

run_detach() {
  (
    set -euo pipefail

    local sniffer_pid=""
    cleanup_detach() {
      if [[ -n "$sniffer_pid" ]] && kill -0 "$sniffer_pid" 2>/dev/null; then
        echo "[INFO] Stopping sniffer pid=$sniffer_pid"
        kill -INT "$sniffer_pid" 2>/dev/null || true
        wait "$sniffer_pid" 2>/dev/null || true
      fi
    }

    trap cleanup_detach EXIT INT TERM

    echo "[INFO] Starting main job: ${MAIN_CMD[*]}"
    "${MAIN_CMD[@]}" &
    local main_pid=$!
    echo "[INFO] Main job pid=$main_pid"

    local -a sniffer_cmd=(
      "$PYTHON_BIN" "$SNIFFER_SCRIPT"
      -p "$main_pid"
      -i "$SAMPLE_INTERVAL"
      -o "$TRACE_OUTPUT"
      --autosave-interval "$AUTOSAVE_INTERVAL"
    )
    if [[ "$ALL_THREADS" -eq 1 ]]; then
      sniffer_cmd+=(--all-threads)
    fi
    if [[ ${#EXTRA_SNIFFER_ARGS[@]} -gt 0 ]]; then
      sniffer_cmd+=("${EXTRA_SNIFFER_ARGS[@]}")
    fi

    echo "[INFO] Starting sniffer: ${sniffer_cmd[*]}"
    "${sniffer_cmd[@]}" &
    sniffer_pid=$!
    echo "[INFO] Sniffer pid=$sniffer_pid"

    {
      echo "main_pid=$main_pid"
      echo "sniffer_pid=$sniffer_pid"
    } > "$PID_FILE"

    set +e
    wait "$main_pid"
    local main_exit_code=$?
    set -e

    echo "[INFO] Main job finished with exit code: $main_exit_code"
    cleanup_detach
    sniffer_pid=""

    exit "$main_exit_code"
  ) > "$DETACH_LOG_FILE" 2>&1 < /dev/null &

  local launcher_pid=$!
  echo "[INFO] Detached mode enabled"
  echo "[INFO] Launcher pid=$launcher_pid"
  echo "[INFO] PID file: $PID_FILE"
  echo "[INFO] Detached log: $DETACH_LOG_FILE"
}

if [[ "$ACTION" != "run" ]]; then
  if [[ "$ACTION" == "status" ]]; then
    run_status
    exit $?
  fi
  if [[ "$ACTION" == "stop" ]]; then
    run_stop
    exit 0
  fi
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[DRY-RUN] Main job: ${MAIN_CMD[*]}"
  if [[ "$DETACH" -eq 1 ]]; then
    echo "[DRY-RUN] Async mode: detach immediately after launch"
  else
    echo "[DRY-RUN] Sync mode: wait main job, then stop sniffer"
  fi
  echo "[DRY-RUN] PID file: $PID_FILE"
  echo "[DRY-RUN] Detached log: $DETACH_LOG_FILE"
  exit 0
fi

if [[ "$DETACH" -eq 1 ]]; then
  run_detach
  exit 0
fi

run_sync
