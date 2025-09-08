#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="/home/hieunt/verl/my_recipe/run_baseline"

run_and_wait() {
  local script_path="$1"
  echo "Submitting $script_path"

  # Execute the script and capture output to parse Ray Job ID if present
  local out job_id status
  out="$(bash "$script_path" 2>&1 | tee /dev/stderr)"

  # Try to parse job submission id from output
  job_id="$(echo "$out" | sed -n 's/.*Job submission id: \([A-Za-z0-9_-]\+\).*/\1/p')"
  if [ -z "${job_id:-}" ]; then
    job_id="$(echo "$out" | awk '/Job submission id:/ {print $4}')"
  fi

  if [ -n "${job_id:-}" ]; then
    echo "Waiting for Ray job $job_id to finish..."
    while true; do
      # Query status; last token is status label
      status="$(ray job status "$job_id" | awk '{print $NF}')"
      case "$status" in
        SUCCEEDED)
          echo "Job $job_id succeeded."
          break
          ;;
        FAILED|CANCELED)
          echo "Job $job_id ended with status: $status"
          exit 1
          ;;
        *)
          sleep 10
          ;;
      esac
    done
  else
    echo "No Ray Job ID detected from output. Assuming script blocks until completion."
  fi
}

# Iterate scripts in lexicographical order and run sequentially
shopt -s nullglob
scripts=("${RUN_DIR}"/*.sh)
if [ ${#scripts[@]} -eq 0 ]; then
  echo "No scripts found in ${RUN_DIR}"
  exit 0
fi

for script in "${scripts[@]}"; do
  run_and_wait "$script"
done

echo "All baseline jobs completed successfully."


