#!/usr/bin/env bash
set -Eeuo pipefail

# Run every experiment behind Fig. 4-8 and Table II-IV.

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

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_ROOT="${OUTPUT_ROOT}/runner_logs"
mkdir -p "${LOG_ROOT}"
MASTER_LOG="${LOG_ROOT}/paper_all_${RUN_ID}.log"
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
from pathlib import Path

path = Path(os.environ["DATA_ROOT"]) / "partition_manifest.json"
manifest = json.loads(path.read_text(encoding="utf-8"))
assert manifest["schema_version"] == 2
assert manifest["datasets"] == ["SMD", "SMAP", "MSL"]
assert manifest["sensors"] == 100
assert manifest["fogs"] == 10
assert manifest["alphas"] == [None]
assert manifest["seeds"] == [42, 43, 44]
observed = {
    (row["dataset"], row["alpha"], row["seed"])
    for row in manifest["partitions"]
}
expected = {
    (dataset, None, seed)
    for dataset in ("SMD", "SMAP", "MSL")
    for seed in (42, 43, 44)
}
assert observed == expected
print(f"Setup manifest validated: {path.resolve()}")
PY

echo "paper_scenarios=scalability,compression,noniid,real"
echo "real_datasets=SMD,SMAP,MSL seeds=42,43,44"
echo "methods=centralized,fedavg,fedprox,hfl-nocoop,hfl-selective,hfl-nearest"
echo "flat_failed_uploads=charge_one_payload_at_SL_MAX_without_rx_or_aggregation"
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
  --suite all
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

RESULT_CSV="${LOG_ROOT}/paper_all_${RUN_ID}_results.csv"
RESULT_JSON="${LOG_ROOT}/paper_all_${RUN_ID}_results.json"
OUTPUT_ROOT="${OUTPUT_ROOT}" \
  QUICK="${QUICK}" \
  RESULT_CSV="${RESULT_CSV}" RESULT_JSON="${RESULT_JSON}" \
  "${PYTHON}" - <<'PY'
import csv
import json
import math
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_ROOT"])
quick = os.environ["QUICK"] == "1"
rows = []

seeds = (42,) if quick else (42, 43, 44)
synthetic_rounds = 2 if quick else 20
real_rounds = 2 if quick else 30
expected = set()

scale_sizes = (50,) if quick else (50, 100, 150, 200)
scale_methods = (
    ("hfl-selective",)
    if quick
    else ("fedprox", "hfl-nocoop", "hfl-selective", "hfl-nearest")
)
for sensors in scale_sizes:
    for method in scale_methods:
        for seed in seeds:
            expected.add(
                ("scalability", "synthetic", sensors, method, seed, 0.05, 1.0, synthetic_rounds)
            )

compression_methods = (
    ("hfl-nocoop",)
    if quick
    else ("fedavg", "fedprox", "hfl-nocoop", "hfl-nearest")
)
for method in compression_methods:
    for rho_s in (0.05, 1.0):
        for seed in seeds:
            expected.add(
                ("compression", "synthetic", 100, method, seed, rho_s, 1.0, synthetic_rounds)
            )

noniid_methods = (
    ("hfl-selective",)
    if quick
    else ("fedprox", "hfl-nocoop", "hfl-selective", "hfl-nearest")
)
for method in noniid_methods:
    for alpha in (0.1, 1.0e4):
        for seed in seeds:
            expected.add(
                ("noniid", "synthetic", 100, method, seed, 0.05, alpha, synthetic_rounds)
            )

real_methods = (
    "centralized",
    "fedavg",
    "fedprox",
    "hfl-nocoop",
    "hfl-selective",
    "hfl-nearest",
)
for dataset in ("SMD", "SMAP", "MSL"):
    for method in real_methods:
        for seed in seeds:
            expected.add(
                ("real", dataset, 100, method, seed, 0.05, None, real_rounds)
            )

def run_key(summary):
    topology = summary.get("topology", {})
    learning = summary.get("learning_config", {})
    alpha = summary.get("partition_alpha")
    return (
        summary.get("scenario"),
        summary.get("dataset"),
        int(topology.get("sensors", -1)),
        summary.get("baseline"),
        int(summary.get("seed", -1)),
        float(learning.get("RHO_S", -1.0)),
        None if alpha is None else float(alpha),
        int(summary.get("rounds", -1)),
    )

for path in root.rglob("summary.json"):
    with path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    key = run_key(summary)
    if key not in expected:
        continue
    final = summary.get("final", {})
    topology = summary.get("topology", {})
    learning = summary.get("learning_config", {})
    required_values = (
        summary.get("best_f1"),
        summary.get("best_pa_f1"),
        summary.get("total_communication_energy_j"),
        summary.get("total_modelled_energy_j"),
        summary.get("total_latency_s"),
        final.get("train_loss"),
        final.get("f1"),
        final.get("pa_f1"),
        final.get("participation"),
    )
    if not all(math.isfinite(float(value)) for value in required_values):
        raise RuntimeError(f"Non-finite completed run: {path}")
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
            "failed_upload_energy_j": summary.get(
                "total_failed_upload_energy_j", 0.0
            ),
            "final_failed_transmission_attempts": final.get(
                "failed_transmission_attempts", 0
            ),
            "total_latency_s": summary.get("total_latency_s"),
            "artifact_dir": str(path.parent),
        }
    )

observed = {
    (
        row["scenario"],
        row["dataset"],
        int(row["sensors"]),
        row["baseline"],
        int(row["seed"]),
        float(row["rho_s"]),
        None if row["partition_alpha"] is None else float(row["partition_alpha"]),
        int(row["rounds"]),
    )
    for row in rows
}
missing = expected - observed
if missing:
    preview = "\n".join(map(str, sorted(missing, key=str)[:10]))
    raise RuntimeError(
        f"Paper matrix incomplete: missing {len(missing)} of {len(expected)} runs\n{preview}"
    )

rows.sort(
    key=lambda row: (
        str(row["scenario"]),
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
    json.dump(rows, handle, indent=2, allow_nan=False)
print(f"Session result index: {csv_path} ({len(rows)} runs)")
print(f"Session result JSON:  {json_path}")
PY

if [[ "${QUICK}" != "1" ]]; then
  "${PYTHON}" -m scripts.paper.plot_all \
    --results "${OUTPUT_ROOT}" \
    --output "${OUTPUT_ROOT}/paper"
fi

echo "RUN COMPLETE"
echo "Raw master log: ${MASTER_LOG}"
echo "Paper artifacts: ${OUTPUT_ROOT}/paper/"
echo "Per-run artifacts: ${OUTPUT_ROOT}/<scenario>/<dataset>/.../"
