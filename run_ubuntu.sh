#!/usr/bin/env bash
set -Eeuo pipefail

# Ubuntu runner for the SMAP/MSL N=100, M=10 real-data experiment matrix.
#
# Examples:
#   bash run_ubuntu.sh all
#   bash run_ubuntu.sh install
#   bash run_ubuntu.sh prepare-data
#   bash run_ubuntu.sh run
#   QUICK=1 WORKERS=4 bash run_ubuntu.sh run

ACTION="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
DATA_ROOT="${DATA_ROOT:-datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results}"
WORKERS="${WORKERS:-}"
QUICK="${QUICK:-0}"
PREPARE_DATA="${PREPARE_DATA:-1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

if (( $# > 1 )); then
  echo "Usage: bash run_ubuntu.sh [install|prepare-data|run|all]" >&2
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

RUN_START_EPOCH="$(date +%s)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_ROOT="${OUTPUT_ROOT}/runner_logs"
mkdir -p "${LOG_ROOT}"
MASTER_LOG="${LOG_ROOT}/${ACTION}_smap_msl_${RUN_ID}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

on_error() {
  local exit_code=$?
  echo "FAILED exit_code=${exit_code} line=${BASH_LINENO[0]}"
  echo "Raw log: ${MASTER_LOG}"
  exit "${exit_code}"
}
trap on_error ERR

echo "action=${ACTION} datasets=SMAP,MSL sensors=100 fogs=10"
echo "baselines=centralized,fedavg,fedprox,hfl-nocoop,hfl-selective,hfl-nearest"
echo "workers=${WORKERS} quick=${QUICK}"
echo "data_root=${DATA_ROOT} output_root=${OUTPUT_ROOT}"
echo "raw_log=${MASTER_LOG}"

install_environment() {
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Missing ${PYTHON_BIN}. Install Python 3.10+ first." >&2
    exit 3
  fi
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Creating virtual environment: ${VENV_DIR}"
    if ! "${PYTHON_BIN}" -m venv "${VENV_DIR}"; then
      echo "venv creation failed. On Ubuntu run:" >&2
      echo "  sudo apt-get update && sudo apt-get install -y python3-venv" >&2
      exit 4
    fi
  fi
  local python="${VENV_DIR}/bin/python"
  "${python}" -m pip install --upgrade pip setuptools wheel
  "${python}" -m pip install --index-url "${TORCH_INDEX_URL}" "torch>=2.0"
  "${python}" -m pip install -r requirements.txt
  "${python}" -m pip check
  "${python}" -m pip freeze > "${LOG_ROOT}/pip_freeze_${RUN_ID}.txt"
  echo "Environment ready: $("${python}" --version)"
}

prepare_benchmarks() {
  local python="${VENV_DIR}/bin/python"
  "${python}" -u -m scripts.prepare_benchmarks \
    --datasets-root "${DATA_ROOT}" \
    --dataset nasa
  DATA_ROOT="${DATA_ROOT}" "${python}" - <<'PY'
import os
from anomaly_detection.data import load_real_benchmark

root = os.environ["DATA_ROOT"]
for name, expected_dim in (
    ("SMAP", 25),
    ("MSL", 55),
):
    bundle = load_real_benchmark(name, root, 100, seed=42)
    assert len(bundle.sensor_train) == 100
    assert bundle.input_dim == expected_dim
    print(
        f"validated {name}: sensors=100 source_entities="
        f"{bundle.details['source_entities']} D={bundle.input_dim} "
        f"train={sum(map(len, bundle.sensor_train.values()))} "
        f"validation={len(bundle.validation_normal)} "
        f"test={len(bundle.test_x)} anomalies={int(bundle.test_y.sum())}"
    )
PY
}

run_experiments() {
  local python="${VENV_DIR}/bin/python"
  local arguments=(
    -u run_experiments.py
    --suite real
    --data-root "${DATA_ROOT}"
    --output-root "${OUTPUT_ROOT}"
    --workers "${WORKERS}"
  )
  if [[ "${QUICK}" == "1" ]]; then
    arguments+=(--quick)
  fi

  # Local sensor models are trained in Python threads. Limit every model to one
  # BLAS/OpenMP thread to avoid workers multiplying the CPU thread count.
  export PYTHONUNBUFFERED=1
  export PYTHONHASHSEED=0
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1

  "${python}" "${arguments[@]}"

  if [[ "${QUICK}" != "1" ]]; then
    "${python}" -m scripts.paper.fig8_real \
      --results "${OUTPUT_ROOT}" \
      --output "${OUTPUT_ROOT}/paper"
    "${python}" -m scripts.paper.tables \
      --only real \
      --results "${OUTPUT_ROOT}" \
      --output "${OUTPUT_ROOT}/paper"
  fi
}

export_session_results() {
  local python="${VENV_DIR}/bin/python"
  local result_csv="${LOG_ROOT}/${ACTION}_smap_msl_${RUN_ID}_results.csv"
  local result_json="${LOG_ROOT}/${ACTION}_smap_msl_${RUN_ID}_results.json"
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
    RUN_START_EPOCH="${RUN_START_EPOCH}" \
    RESULT_CSV="${result_csv}" RESULT_JSON="${result_json}" \
    "${python}" - <<'PY'
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
        str(row["scenario"]),
        str(row["dataset"]),
        int(row["sensors"] or 0),
        str(row["baseline"]),
        float(row["rho_s"] or 0.0),
        float(row["partition_alpha"] or 0.0),
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
}

case "${ACTION}" in
  install)
    install_environment
    ;;
  prepare-data)
    install_environment
    prepare_benchmarks
    ;;
  run)
    install_environment
    if [[ "${PREPARE_DATA}" == "1" ]]; then
      prepare_benchmarks
    fi
    run_experiments
    export_session_results
    ;;
  all)
    install_environment
    prepare_benchmarks
    run_experiments
    export_session_results
    ;;
  *)
    echo "Invalid action: ${ACTION}" >&2
    echo "Choose: install, prepare-data, run, all" >&2
    exit 2
    ;;
esac

echo "COMPLETE"
echo "Raw master log: ${MASTER_LOG}"
echo "Per-run artifacts: ${OUTPUT_ROOT}/real/<dataset>/N_100_M_10/<baseline>/.../"
