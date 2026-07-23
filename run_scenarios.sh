#!/usr/bin/env bash
set -Eeuo pipefail

# Run the fixed SMAP/MSL experiment matrix. Execute setup_ubuntu.sh first.

VENV_DIR="${VENV_DIR:-.venv}"
DATA_ROOT="${DATA_ROOT:-datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results}"
WORKERS="${WORKERS:-}"
TORCH_THREADS="${TORCH_THREADS:-1}"
QUICK="${QUICK:-0}"

if (( $# != 0 )); then
  echo "Usage: bash run_scenarios.sh" >&2
  exit 2
fi

if [[ -z "${WORKERS}" ]]; then
  CPU_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 1)"
  if (( CPU_COUNT > 8 )); then
    WORKERS=8
  else
    WORKERS="${CPU_COUNT}"
  fi
fi
CENTRALIZED_TORCH_THREADS="${CENTRALIZED_TORCH_THREADS:-${WORKERS}}"

RUN_START_EPOCH="$(date +%s)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_ROOT="${OUTPUT_ROOT}/runner_logs"
mkdir -p "${LOG_ROOT}"
MASTER_LOG="${LOG_ROOT}/smap_msl_${RUN_ID}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

on_error() {
  local exit_code=$?
  echo "RUN FAILED exit_code=${exit_code} line=${BASH_LINENO[0]}"
  echo "Raw log: ${MASTER_LOG}"
  exit "${exit_code}"
}
trap on_error ERR

PYTHON="${VENV_DIR}/bin/python"
MANIFEST="${DATA_ROOT}/partition_manifest.json"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing ${PYTHON}. Run: bash setup_ubuntu.sh" >&2
  exit 3
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing ${MANIFEST}. Run: bash setup_ubuntu.sh" >&2
  exit 4
fi

DATA_ROOT="${DATA_ROOT}" "${PYTHON}" - <<'PY'
import json
import os
from itertools import product
from pathlib import Path

path = Path(os.environ["DATA_ROOT"]) / "partition_manifest.json"
manifest = json.loads(path.read_text(encoding="utf-8"))
assert manifest["datasets"] == ["SMAP", "MSL"]
assert manifest["sensors"] == 100
assert manifest["fogs"] == 10
assert manifest["alphas"] == [0.1, 1.0e4]
assert manifest["seeds"] == [42, 43, 44]
observed = {
    (row["dataset"], row["alpha"], row["seed"])
    for row in manifest["partitions"]
}
expected = set(product(("SMAP", "MSL"), (0.1, 1.0e4), (42, 43, 44)))
assert observed == expected
print(f"Setup manifest validated: {path.resolve()}")
PY

echo "datasets=SMAP,MSL sensors=100 fogs=10"
echo "dirichlet_alpha=0.1,10000 seeds=42,43,44"
echo "baselines=centralized,fedavg,fedprox,hfl-nocoop,hfl-selective,hfl-nearest"
echo "workers=${WORKERS} torch_threads_per_worker=${TORCH_THREADS} centralized_torch_threads=${CENTRALIZED_TORCH_THREADS} backend=process quick=${QUICK}"
echo "raw_log=${MASTER_LOG}"

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ARGUMENTS=(
  -u run_experiments.py
  --suite real
  --data-root "${DATA_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --workers "${WORKERS}"
  --torch-threads "${TORCH_THREADS}"
  --centralized-torch-threads "${CENTRALIZED_TORCH_THREADS}"
  --parallel-backend process
)
if [[ "${QUICK}" == "1" ]]; then
  ARGUMENTS+=(--quick)
fi
"${PYTHON}" "${ARGUMENTS[@]}"

if [[ "${QUICK}" != "1" ]]; then
  "${PYTHON}" -m scripts.paper.fig8_real \
    --results "${OUTPUT_ROOT}" \
    --output "${OUTPUT_ROOT}/paper"
  "${PYTHON}" -m scripts.paper.tables \
    --only real \
    --results "${OUTPUT_ROOT}" \
    --output "${OUTPUT_ROOT}/paper"
fi

RESULT_CSV="${LOG_ROOT}/smap_msl_${RUN_ID}_results.csv"
RESULT_JSON="${LOG_ROOT}/smap_msl_${RUN_ID}_results.json"
OUTPUT_ROOT="${OUTPUT_ROOT}" \
  RUN_START_EPOCH="${RUN_START_EPOCH}" \
  RESULT_CSV="${RESULT_CSV}" RESULT_JSON="${RESULT_JSON}" \
  "${PYTHON}" - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_ROOT"])
started_at = int(os.environ["RUN_START_EPOCH"])
rows = []

for path in root.rglob("summary.json"):
    if path.stat().st_mtime < started_at:
        continue
    with path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("scenario") != "real":
        continue
    if summary.get("dataset") not in {"SMAP", "MSL"}:
        continue
    final = summary.get("final", {})
    topology = summary.get("topology", {})
    learning = summary.get("learning_config", {})
    rows.append(
        {
            "scenario": summary.get("scenario"),
            "dataset": summary.get("dataset"),
            "baseline": summary.get("baseline"),
            "seed": summary.get("seed"),
            "sensors": topology.get("sensors"),
            "fogs": topology.get("fogs"),
            "rounds": summary.get("rounds"),
            "rho_s": learning.get("RHO_S"),
            "partition_alpha": summary.get("partition_alpha"),
            "best_f1": summary.get("best_f1"),
            "best_pa_f1": summary.get("best_pa_f1"),
            "final_train_loss": final.get("train_loss"),
            "final_f1": final.get("f1"),
            "final_pa_f1": final.get("pa_f1"),
            "final_participation": final.get("participation"),
            "communication_energy_j": summary.get("total_communication_energy_j"),
            "total_modelled_energy_j": summary.get("total_modelled_energy_j"),
            "total_latency_s": summary.get("total_latency_s"),
            "artifact_dir": str(path.parent),
        }
    )

rows.sort(
    key=lambda row: (
        str(row["dataset"]),
        float(row["partition_alpha"] or 0.0),
        str(row["baseline"]),
        int(row["seed"] or 0),
    )
)
csv_path = Path(os.environ["RESULT_CSV"])
json_path = Path(os.environ["RESULT_JSON"])
fieldnames = list(rows[0]) if rows else [
    "scenario",
    "dataset",
    "baseline",
    "seed",
    "artifact_dir",
]
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
with json_path.open("w", encoding="utf-8") as handle:
    json.dump(rows, handle, indent=2)
print(f"Session result index: {csv_path} ({len(rows)} runs)")
print(f"Session result JSON:  {json_path}")
PY

echo "RUN COMPLETE"
echo "Raw master log: ${MASTER_LOG}"
echo "Per-run artifacts: ${OUTPUT_ROOT}/real/<dataset>/N_100_M_10/<baseline>/.../"
