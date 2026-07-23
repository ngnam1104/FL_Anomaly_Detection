"""Synthetic and real benchmark data preparation."""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import torch


@dataclass
class DataBundle:
    name: str
    input_dim: int
    sensor_train: Dict[int, torch.Tensor]
    validation_normal: torch.Tensor
    test_x: torch.Tensor
    test_y: np.ndarray
    partition_alpha: float | None = None
    details: dict = field(default_factory=dict)

    def raw_bits_by_sensor(self, bits_per_value: int = 32) -> Dict[int, int]:
        """Raw normal-training payload used by the centralised oracle."""

        return {
            sensor_id: int(samples.numel() * bits_per_value)
            for sensor_id, samples in self.sensor_train.items()
        }


def _standardise(
    train: np.ndarray, validation: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, (validation - mean) / std, (test - mean) / std


def _partition_rows(data: np.ndarray, sensor_count: int, seed: int) -> Dict[int, torch.Tensor]:
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(data))
    shards = np.array_split(order, sensor_count)
    return {
        sensor_id: torch.as_tensor(data[index], dtype=torch.float32)
        for sensor_id, index in enumerate(shards)
    }


def _partition_entity_sequences(
    parts: list[np.ndarray],
    entity_ids: list[str],
    sensor_count: int,
) -> tuple[list[np.ndarray], list[str]]:
    """Split entity sequences into contiguous, non-overlapping sensor shards."""

    if sensor_count < len(parts):
        raise ValueError(
            f"Cannot map {len(parts)} source entities to only {sensor_count} sensors"
        )
    lengths = np.asarray([len(part) for part in parts], dtype=np.int64)
    if sensor_count > int(lengths.sum()):
        raise ValueError(
            f"Requested {sensor_count} sensors but only {int(lengths.sum())} "
            "normal training rows are available"
        )
    allocations = np.ones(len(parts), dtype=np.int64)
    for _ in range(sensor_count - len(parts)):
        eligible = allocations < lengths
        scores = np.where(eligible, lengths / allocations, -np.inf)
        entity_index = int(np.argmax(scores))
        if not np.isfinite(scores[entity_index]):
            raise ValueError("Unable to create non-empty sensor partitions")
        allocations[entity_index] += 1

    sensor_parts: list[np.ndarray] = []
    sensor_entities: list[str] = []
    for part, entity_id, shard_count in zip(parts, entity_ids, allocations):
        shards = np.array_split(part, int(shard_count))
        if any(len(shard) == 0 for shard in shards):
            raise ValueError(f"{entity_id}: empty sensor shard generated")
        sensor_parts.extend(shards)
        sensor_entities.extend([entity_id] * len(shards))
    return sensor_parts, sensor_entities


def make_synthetic(
    sensor_count: int,
    *,
    feature_dim: int = 32,
    samples_per_sensor: int = 128,
    validation_samples: int = 1024,
    test_samples: int = 2048,
    anomaly_fraction: float = 0.12,
    heterogeneity: float = 0.35,
    dirichlet_alpha: float = 1.0,
    regimes: int = 4,
    seed: int = 42,
) -> DataBundle:
    """Create normal data with Dirichlet-controlled client heterogeneity."""

    rng = np.random.RandomState(seed)
    latent_dim = min(8, feature_dim)
    projection = rng.normal(0.0, 1.0 / np.sqrt(latent_dim), (latent_dim, feature_dim))
    regime_offsets = rng.normal(0.0, heterogeneity, (regimes, feature_dim))
    sensor_train: Dict[int, torch.Tensor] = {}
    entropies = []
    for sensor_id in range(sensor_count):
        proportions = rng.dirichlet(np.full(regimes, dirichlet_alpha))
        entropies.append(
            float(-np.sum(proportions * np.log(proportions + 1e-12)) / np.log(regimes))
        )
        regime_ids = rng.choice(regimes, size=samples_per_sensor, p=proportions)
        latent = rng.normal(size=(samples_per_sensor, latent_dim))
        normal = np.tanh(latent @ projection + regime_offsets[regime_ids])
        normal += rng.normal(0.0, 0.04, normal.shape)
        sensor_train[sensor_id] = torch.as_tensor(normal, dtype=torch.float32)

    val_latent = rng.normal(size=(validation_samples, latent_dim))
    val_regimes = rng.randint(0, regimes, size=validation_samples)
    validation = np.tanh(val_latent @ projection + regime_offsets[val_regimes])
    validation += rng.normal(0.0, 0.04, validation.shape)

    test_latent = rng.normal(size=(test_samples, latent_dim))
    test_regimes = rng.randint(0, regimes, size=test_samples)
    test = np.tanh(test_latent @ projection + regime_offsets[test_regimes])
    test += rng.normal(0.0, 0.04, test.shape)
    labels = np.zeros(test_samples, dtype=np.int64)
    target_anomalies = max(1, int(test_samples * anomaly_fraction))
    written = 0
    while written < target_anomalies:
        length = min(rng.randint(4, 18), target_anomalies - written)
        start = rng.randint(0, max(1, test_samples - length))
        indices = np.arange(start, start + length)
        fresh = indices[labels[indices] == 0]
        if not len(fresh):
            continue
        affected = rng.choice(feature_dim, size=max(1, feature_dim // 4), replace=False)
        test[np.ix_(fresh, affected)] += rng.choice([-1.0, 1.0]) * rng.uniform(2.5, 4.5)
        labels[fresh] = 1
        written += len(fresh)

    return DataBundle(
        name="synthetic",
        input_dim=feature_dim,
        sensor_train=sensor_train,
        validation_normal=torch.as_tensor(validation, dtype=torch.float32),
        test_x=torch.as_tensor(test, dtype=torch.float32),
        test_y=labels,
        partition_alpha=float(dirichlet_alpha),
        details={
            "samples_per_sensor": int(samples_per_sensor),
            "validation_samples": int(validation_samples),
            "test_samples": int(test_samples),
            "anomaly_fraction": float(anomaly_fraction),
            "regimes": int(regimes),
            "mean_normalised_client_entropy": float(np.mean(entropies)),
        },
    )


def _load_processed(root: Path, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates = [
        (
            root / f"{name}_train.npy",
            root / f"{name}_test.npy",
            root / f"{name}_test_label.npy",
        ),
        (root / name / "train.npy", root / name / "test.npy", root / name / "labels.npy"),
    ]
    for train_path, test_path, label_path in candidates:
        if all(path.exists() for path in (train_path, test_path, label_path)):
            return np.load(train_path), np.load(test_path), np.load(label_path)
    raise FileNotFoundError(
        f"Processed {name} arrays not found under {root}. Expected "
        f"{name}_train.npy, {name}_test.npy and {name}_test_label.npy."
    )


def _load_smd_raw_bundle(root: Path, machine_limit: int) -> DataBundle:
    """Keep each selected SMD machine as one physical sensor/client."""

    base = root / "SMD"
    train_files = sorted((base / "train").glob("*.txt"))
    if machine_limit:
        train_files = train_files[:machine_limit]
    if not train_files:
        raise FileNotFoundError(f"No SMD machine files found under {base / 'train'}")
    train_parts, validation_parts, tests, labels = [], [], [], []
    for train_path in train_files:
        test_path = base / "test" / train_path.name
        label_path = base / "test_label" / train_path.name
        machine_train = np.loadtxt(train_path, delimiter=",").astype(np.float32)
        split = max(1, int(0.9 * len(machine_train)))
        train_parts.append(machine_train[:split])
        validation_parts.append(machine_train[split:])
        machine_test = np.loadtxt(test_path, delimiter=",").astype(np.float32)
        machine_labels = np.loadtxt(label_path, delimiter=",").reshape(-1)
        if len(machine_test) != len(machine_labels):
            raise ValueError(
                f"SMD test/label length mismatch for {train_path.name}: "
                f"{len(machine_test)} != {len(machine_labels)}"
            )
        tests.append(machine_test)
        labels.append(machine_labels)
    pooled_train = np.concatenate(train_parts)
    mean = pooled_train.mean(axis=0, keepdims=True)
    std = pooled_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    sensor_train = {
        sensor_id: torch.as_tensor((part - mean) / std, dtype=torch.float32)
        for sensor_id, part in enumerate(train_parts)
    }
    validation = np.concatenate(validation_parts)
    test = np.concatenate(tests)
    return DataBundle(
        name="SMD",
        input_dim=pooled_train.shape[1],
        sensor_train=sensor_train,
        validation_normal=torch.as_tensor((validation - mean) / std, dtype=torch.float32),
        test_x=torch.as_tensor((test - mean) / std, dtype=torch.float32),
        test_y=np.concatenate(labels).astype(np.int64),
        details={"entities": len(sensor_train), "source_layout": "raw-smd"},
    )


def _find_telemanom_root(root: Path) -> Path:
    candidates = (
        root / "telemanom",
        root / "Telemanom",
        root / "data" / "data",
        root / "data",
        root,
    )
    for candidate in candidates:
        if (
            (candidate / "labeled_anomalies.csv").exists()
            and (candidate / "train").is_dir()
            and (candidate / "test").is_dir()
        ):
            return candidate
    raise FileNotFoundError(
        "Telemanom data not found. Expected labeled_anomalies.csv plus "
        "train/*.npy and test/*.npy under datasets/telemanom."
    )


def _load_telemanom_bundle(
    root: Path, dataset: str, sensor_count: int
) -> DataBundle:
    """Load each SMAP/MSL telemetry channel as one physical FL client."""

    base = _find_telemanom_root(root)
    with (base / "labeled_anomalies.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        rows = list(csv.DictReader(handle))
    channel_key = "chan_id" if rows and "chan_id" in rows[0] else "channel_id"
    selected = sorted(
        (row for row in rows if row.get("spacecraft", "").upper() == dataset),
        key=lambda row: row[channel_key],
    )
    if sensor_count <= 0:
        raise ValueError("sensor_count must be positive")
    if sensor_count < len(selected):
        selected = selected[:sensor_count]
    if not selected:
        raise ValueError(f"No {dataset} channels found in labeled_anomalies.csv")

    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    tests: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    channel_ids: list[str] = []
    input_dim: int | None = None
    anomaly_segments = 0
    for row in selected:
        channel_id = row[channel_key]
        train = np.load(base / "train" / f"{channel_id}.npy").astype(np.float32)
        test = np.load(base / "test" / f"{channel_id}.npy").astype(np.float32)
        if train.ndim == 1:
            train = train[:, None]
        if test.ndim == 1:
            test = test[:, None]
        if train.shape[1] != test.shape[1]:
            raise ValueError(
                f"{channel_id}: train/test dimensions differ "
                f"({train.shape[1]} != {test.shape[1]})"
            )
        if input_dim is None:
            input_dim = int(train.shape[1])
        elif train.shape[1] != input_dim:
            raise ValueError(
                f"{dataset} channel {channel_id} has D={train.shape[1]}, "
                f"expected D={input_dim}"
            )
        split = min(len(train) - 1, max(1, int(0.9 * len(train))))
        if split <= 0:
            raise ValueError(f"{channel_id}: training sequence is too short")
        train_parts.append(train[:split])
        validation_parts.append(train[split:])
        tests.append(test)
        channel_labels = np.zeros(len(test), dtype=np.int64)
        sequences = ast.literal_eval(row["anomaly_sequences"])
        for start, end in sequences:
            lo = max(0, int(start))
            hi = min(len(test) - 1, int(end))
            if lo <= hi:
                channel_labels[lo : hi + 1] = 1
                anomaly_segments += 1
        labels.append(channel_labels)
        channel_ids.append(channel_id)

    expected_dim = {"SMAP": 25, "MSL": 55}[dataset]
    if input_dim != expected_dim:
        raise ValueError(
            f"{dataset} should have D={expected_dim}, found D={input_dim}"
        )
    pooled_train = np.concatenate(train_parts)
    mean = pooled_train.mean(axis=0, keepdims=True)
    std = pooled_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    sensor_parts, sensor_channel_ids = _partition_entity_sequences(
        train_parts, channel_ids, sensor_count
    )
    sensor_train = {
        sensor_id: torch.as_tensor((part - mean) / std, dtype=torch.float32)
        for sensor_id, part in enumerate(sensor_parts)
    }
    validation = np.concatenate(validation_parts)
    test = np.concatenate(tests)
    return DataBundle(
        name=dataset,
        input_dim=expected_dim,
        sensor_train=sensor_train,
        validation_normal=torch.as_tensor(
            (validation - mean) / std, dtype=torch.float32
        ),
        test_x=torch.as_tensor((test - mean) / std, dtype=torch.float32),
        test_y=np.concatenate(labels),
        details={
            "entities": len(sensor_train),
            "source_entities": len(channel_ids),
            "channel_ids": channel_ids,
            "sensor_channel_ids": sensor_channel_ids,
            "anomaly_segments": anomaly_segments,
            "source_layout": "telemanom-contiguous-sensor-shards",
        },
    )


def load_real_benchmark(
    name: str,
    data_root: str | Path,
    sensor_count: int,
    *,
    seed: int = 42,
) -> DataBundle:
    """Load SMD/SMAP/MSL from standard processed NumPy arrays or raw SMD."""

    dataset = name.upper()
    root = Path(data_root)
    if dataset == "SMD" and (root / "SMD" / "train").is_dir():
        return _load_smd_raw_bundle(root, sensor_count)
    if dataset in {"SMAP", "MSL"}:
        try:
            return _load_telemanom_bundle(root, dataset, sensor_count)
        except FileNotFoundError:
            pass
    try:
        train, test, labels = _load_processed(root, dataset)
    except FileNotFoundError:
        if dataset != "SMD":
            raise
        return _load_smd_raw_bundle(root, sensor_count)
    train = np.asarray(train, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    if train.ndim == 1:
        train = train[:, None]
    if test.ndim == 1:
        test = test[:, None]
    split = max(1, int(0.9 * len(train)))
    train_data, validation = train[:split], train[split:]
    train_data, validation, test = _standardise(train_data, validation, test)
    return DataBundle(
        name=dataset,
        input_dim=train_data.shape[1],
        sensor_train=_partition_rows(train_data, sensor_count, seed),
        validation_normal=torch.as_tensor(validation, dtype=torch.float32),
        test_x=torch.as_tensor(test, dtype=torch.float32),
        test_y=labels,
        details={
            "entities": int(sensor_count),
            "source_layout": "processed-numpy",
        },
    )
