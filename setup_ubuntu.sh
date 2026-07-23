#!/usr/bin/env bash
set -Eeuo pipefail

# One-time Ubuntu setup: Python environment, all three real benchmarks, and
# deterministic validation for every paper seed.

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
DATA_ROOT="${DATA_ROOT:-datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
INSTALL_SYSTEM_VENV="${INSTALL_SYSTEM_VENV:-0}"

if (( $# != 0 )); then
  echo "Usage: bash setup_ubuntu.sh" >&2
  exit 2
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_ROOT="${OUTPUT_ROOT}/setup_logs"
mkdir -p "${LOG_ROOT}"
SETUP_LOG="${LOG_ROOT}/setup_${RUN_ID}.log"
exec > >(tee -a "${SETUP_LOG}") 2>&1

on_error() {
  local exit_code=$?
  echo "SETUP FAILED exit_code=${exit_code} line=${BASH_LINENO[0]}"
  echo "Raw log: ${SETUP_LOG}"
  exit "${exit_code}"
}
trap on_error ERR

echo "setup real_datasets=SMD,SMAP,MSL sensors=100 fogs=10"
echo "real_partition=contiguous_source_shards seeds=42,43,44"
echo "venv=${VENV_DIR} data_root=${DATA_ROOT}"
echo "raw_log=${SETUP_LOG}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Missing ${PYTHON_BIN}. Install Python 3.10+ first." >&2
  exit 3
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Creating virtual environment: ${VENV_DIR}"
  if ! "${PYTHON_BIN}" -m venv "${VENV_DIR}"; then
    PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    VENV_PACKAGE="python${PYTHON_VERSION}-venv"
    if [[ "${INSTALL_SYSTEM_VENV}" == "1" ]] && command -v apt-get >/dev/null 2>&1; then
      echo "Installing missing system package: ${VENV_PACKAGE}"
      sudo apt-get update
      sudo apt-get install -y "${VENV_PACKAGE}"
      "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    else
      echo "venv creation failed. On Debian/Ubuntu run:" >&2
      echo "  sudo apt-get update && sudo apt-get install -y ${VENV_PACKAGE}" >&2
      echo "Or let this script install it interactively:" >&2
      echo "  INSTALL_SYSTEM_VENV=1 bash setup_ubuntu.sh" >&2
      exit 4
    fi
  fi
fi

PYTHON="${VENV_DIR}/bin/python"
"${PYTHON}" -m pip install --upgrade pip setuptools wheel
"${PYTHON}" -m pip install --index-url "${TORCH_INDEX_URL}" "torch>=2.0"
"${PYTHON}" -m pip install -r requirements.txt
"${PYTHON}" -m pip check
"${PYTHON}" -m pip freeze > "${LOG_ROOT}/pip_freeze_${RUN_ID}.txt"
echo "Environment ready: $("${PYTHON}" --version)"

"${PYTHON}" -u -m scripts.prepare_benchmarks \
  --datasets-root "${DATA_ROOT}" \
  --dataset all

DATA_ROOT="${DATA_ROOT}" "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

from anomaly_detection.data import load_real_benchmark

root = Path(os.environ["DATA_ROOT"])
datasets = (("SMD", 38), ("SMAP", 25), ("MSL", 55))
seeds = (42, 43, 44)
partitions = []

for name, expected_dim in datasets:
    for seed in seeds:
        bundle = load_real_benchmark(name, root, 100, seed=seed)
        sizes = [len(samples) for samples in bundle.sensor_train.values()]
        assert len(sizes) == 100
        assert min(sizes) > 0
        assert bundle.input_dim == expected_dim
        source_entities = bundle.details["source_entities"]
        assert source_entities > 0
        if name == "SMD":
            assert source_entities == 10
        partitions.append(
            {
                "dataset": name,
                "alpha": None,
                "seed": seed,
                "sensors": 100,
                "fogs": 10,
                "input_dim": bundle.input_dim,
                "source_entities": source_entities,
                "train_rows": sum(sizes),
                "min_sensor_rows": min(sizes),
                "max_sensor_rows": max(sizes),
            }
        )
        print(
            f"validated {name}: seed={seed} N=100 M=10 D={bundle.input_dim} "
            f"sources={source_entities} rows={sum(sizes)} "
            f"min/max={min(sizes)}/{max(sizes)}"
        )

manifest = {
    "schema_version": 2,
    "datasets": ["SMD", "SMAP", "MSL"],
    "sensors": 100,
    "fogs": 10,
    "alphas": [None],
    "seeds": list(seeds),
    "partition_scheme": "contiguous-entity-shards",
    "partitions": partitions,
}
manifest_path = root / "partition_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Partition manifest: {manifest_path.resolve()}")
PY

echo "SETUP COMPLETE"
echo "Raw setup log: ${SETUP_LOG}"
echo "Partition manifest: ${DATA_ROOT}/partition_manifest.json"
echo "Next: bash run_scenarios.sh"
