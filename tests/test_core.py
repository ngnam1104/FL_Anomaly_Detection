from __future__ import annotations

import copy
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from anomaly_detection.compression import ErrorFeedbackTopK
from anomaly_detection.data import (
    _partition_entity_sequences,
    load_real_benchmark,
    make_synthetic,
)
from anomaly_detection.model import Autoencoder
from anomaly_detection.simulator import AnomalyFLSimulator, _train_sensor_worker
from config.settings import (
    acoustic_cfg,
    energy_cfg,
    learning_cfg,
    network_cfg,
    table_ii_rows,
)
from anomaly_detection.hfl_rules import select_cooperation
from physics_models.communication import (
    min_source_level,
    shannon_capacity,
    thorp_absorption,
    transmission_loss,
    wenz_noise_level,
)
from physics_models.energy import acoustic_power_watts, e_comp, e_rx, e_tx
from physics_models.latency import round_latency
from physics_models.topology import Topology3D


torch.set_num_threads(1)


def test_paper_autoencoder_parameter_count():
    assert Autoencoder(32).parameter_count == 1352


def test_acoustic_equations_match_fedkdl_regression_values():
    """These constants lock the unchanged FedKDL Thorp/Wenz/SNR equations."""

    assert thorp_absorption(12.0) == pytest.approx(1.6447725762943222)
    assert transmission_loss(750.0, 12.0, 1.5) == pytest.approx(
        44.359498383096245
    )
    assert wenz_noise_level(12.0, 4000.0, 5.0, 0.5) == pytest.approx(
        80.63887096407171
    )
    assert min_source_level(
        750.0, 12.0, 4000.0, 10.0, 2.0, 1.5, 5.0, 0.5
    ) == pytest.approx(136.99836934716797)
    rate = shannon_capacity(4000.0, 10.0)
    assert rate == pytest.approx(13837.726474549188)
    assert acoustic_power_watts(140.0) == pytest.approx(0.0008173249180071006)
    assert e_tx(1292, rate, 140.0, 0.25, 0.05) == pytest.approx(
        0.004973644717059844
    )
    assert e_rx(1292, rate, 0.03) == pytest.approx(0.0028010381670203334)


def test_table_ii_runtime_values():
    assert network_cfg.SENSOR_DEPTH == (500.0, 1000.0)
    assert network_cfg.FOG_DEPTH == (100.0, 400.0)
    assert acoustic_cfg.SL_MAX == 140.0
    assert energy_cfg.E_INIT == 500.0
    assert energy_cfg.P_C_TX == 0.05
    assert energy_cfg.P_C_RX == 0.03
    assert energy_cfg.EPSILON_OP == 2.8e-10
    configured_flops_per_second = (
        energy_cfg.F_CPU * energy_cfg.N_CORES * energy_cfg.FLOPS_PER_CYCLE
    )
    assert e_comp(
        configured_flops_per_second, energy_cfg.EPSILON_OP
    ) == pytest.approx(10.08)
    assert learning_cfg.LOCAL_EPOCHS == 5
    assert learning_cfg.LOCAL_LR == 0.01
    assert learning_cfg.MAX_GRAD_NORM == 5.0
    assert len(table_ii_rows()) == 23


def test_topk_payload_and_error_feedback():
    compressor = ErrorFeedbackTopK(100, ratio=0.05)
    update = torch.linspace(-1.0, 1.0, 100)
    first = compressor.compress(update)
    assert first.indices.numel() == 5
    assert first.payload_bits == 75
    assert torch.count_nonzero(first.decompress()) <= 5
    assert torch.linalg.vector_norm(compressor.error) > 0


def test_paper_payload_sizes():
    compressed = ErrorFeedbackTopK(1352, ratio=0.05).compress(torch.ones(1352))
    assert compressed.indices.numel() == 68
    assert compressed.payload_bits == 1292
    assert 1352 * 32 == 43264


def test_local_sgd_stays_finite_with_real_benchmark_scale_outlier():
    model = Autoencoder(25)
    samples = torch.randn(64, 25)
    samples[0, 0] = 356.0
    result = _train_sensor_worker(
        {
            "sensor_id": 7,
            "global_state": model.state_dict(),
            "samples": samples,
            "input_dim": 25,
            "hidden_dims": (16, 8),
            "local_lr": 0.01,
            "max_grad_norm": 5.0,
            "local_epochs": 5,
            "batch_size": 32,
            "fedprox_mu": 0.02,
            "use_prox": False,
            "round_index": 1,
            "train_seed": 42,
        }
    )
    assert np.isfinite(result["loss"])
    assert all(torch.isfinite(value).all() for value in result["delta_state"].values())


def test_local_training_limits_in_flight_tensor_tasks_to_worker_count():
    class TrackedFuture(Future):
        def __init__(self, owner, result):
            super().__init__()
            self.owner = owner
            self.consumed = False
            self.set_result(result)

        def result(self, timeout=None):
            if not self.consumed:
                self.owner.outstanding -= 1
                self.consumed = True
            return super().result(timeout)

    class ImmediatePool:
        def __init__(self):
            self.outstanding = 0
            self.max_outstanding = 0

        def submit(self, _function, task):
            self.outstanding += 1
            self.max_outstanding = max(self.max_outstanding, self.outstanding)
            return TrackedFuture(
                self,
                {
                    "sensor_id": task["sensor_id"],
                    "samples": 1,
                    "loss": 0.0,
                    "payload_bits": 0,
                },
            )

    simulator = object.__new__(AnomalyFLSimulator)
    simulator.workers = 3
    simulator.global_model = torch.nn.Linear(1, 1)
    simulator.log = Mock()
    simulator._training_task = (
        lambda sensor_id, _global_state, _round_index: {"sensor_id": sensor_id}
    )
    simulator._finalise_local_update = (
        lambda raw_result, compress_update: raw_result
    )
    pool = ImmediatePool()

    results = simulator._parallel_local_training(list(range(20)), 1, pool)

    assert [result["sensor_id"] for result in results] == list(range(20))
    assert pool.max_outstanding == simulator.workers
    assert pool.outstanding == 0


def test_round_latency_uses_slowest_link_plus_compute():
    assert round_latency([0.4, 1.7, 0.8], 0.3) == pytest.approx(2.0)


def test_only_fog_nodes_move():
    net = copy.deepcopy(network_cfg)
    net.N_SENSORS = 8
    net.M_FOGS = 3
    topology = Topology3D(net, acoustic_cfg, seed=7)
    sensors_before = topology.sensor_positions.copy()
    fogs_before = topology.fog_positions.copy()
    topology.step_mobile_fogs()
    np.testing.assert_array_equal(topology.sensor_positions, sensors_before)
    assert np.any(topology.fog_positions != fogs_before)


def test_selective_cooperation_uses_larger_nearby_cluster():
    graph = {
        ("fog", 0, "fog", 1): SimpleNamespace(distance=10.0),
        ("fog", 1, "fog", 0): SimpleNamespace(distance=10.0),
        ("fog", 0, "fog", 2): SimpleNamespace(distance=20.0),
        ("fog", 2, "fog", 0): SimpleNamespace(distance=20.0),
        ("fog", 1, "fog", 2): SimpleNamespace(distance=30.0),
        ("fog", 2, "fog", 1): SimpleNamespace(distance=30.0),
    }
    partners = select_cooperation("selective", {0: 1, 1: 5, 2: 6}, graph)
    assert partners == {0: 1}
    for receiver, donor in partners.items():
        assert {0: 1, 1: 5, 2: 6}[donor] > {0: 1, 1: 5, 2: 6}[receiver]


def test_dirichlet_alpha_controls_client_heterogeneity():
    non_iid = make_synthetic(
        40,
        samples_per_sensor=8,
        validation_samples=16,
        test_samples=32,
        dirichlet_alpha=0.1,
        seed=17,
    )
    near_iid = make_synthetic(
        40,
        samples_per_sensor=8,
        validation_samples=16,
        test_samples=32,
        dirichlet_alpha=1.0e4,
        seed=17,
    )
    assert (
        non_iid.details["mean_normalised_client_entropy"]
        < near_iid.details["mean_normalised_client_entropy"]
    )


@pytest.mark.parametrize(
    ("entity_count", "expected_counts"),
    (
        (10, {10: 10}),
        (54, {1: 8, 2: 46}),
        (55, {1: 10, 2: 45}),
        (27, {3: 8, 4: 19}),
    ),
)
def test_paper_real_partitions_use_balanced_contiguous_shards(
    entity_count, expected_counts
):
    parts = [
        np.arange(500 + entity_id, dtype=np.float32)[:, None]
        for entity_id in range(entity_count)
    ]
    entity_ids = [f"entity-{index}" for index in range(entity_count)]

    shards, shard_entities = _partition_entity_sequences(
        parts, entity_ids, 100, seed=42
    )

    observed = {
        count: list(shard_entities).count(count)
        for count in set(shard_entities)
    }
    distribution = {
        shard_count: sum(value == shard_count for value in observed.values())
        for shard_count in set(observed.values())
    }
    assert distribution == expected_counts
    assert len(shards) == 100
    assert all(len(shard) > 0 for shard in shards)


def test_telemanom_keeps_one_channel_per_client(tmp_path):
    base = tmp_path / "telemanom"
    (base / "train").mkdir(parents=True)
    (base / "test").mkdir()
    rows = [
        'chan_id,spacecraft,anomaly_sequences,class,num_values',
        'A-1,SMAP,"[[2, 4]]",[point],10',
        'A-2,SMAP,[],[point],10',
    ]
    (base / "labeled_anomalies.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    for channel_index, channel_id in enumerate(("A-1", "A-2")):
        train = np.full((20, 25), channel_index, dtype=np.float32)
        test = np.full((10, 25), channel_index, dtype=np.float32)
        np.save(base / "train" / f"{channel_id}.npy", train)
        np.save(base / "test" / f"{channel_id}.npy", test)

    bundle = load_real_benchmark("SMAP", tmp_path, 2)
    assert bundle.input_dim == 25
    assert len(bundle.sensor_train) == 2
    assert bundle.details["channel_ids"] == ["A-1", "A-2"]
    assert bundle.details["source_layout"] == "telemanom-per-channel"
    assert bundle.details["partition_scheme"] == "contiguous-entity-shards"
    assert len(bundle.test_y) == 20
    assert int(bundle.test_y.sum()) == 3


def test_telemanom_merges_duplicate_channel_metadata_rows(tmp_path):
    base = tmp_path / "telemanom"
    (base / "train").mkdir(parents=True)
    (base / "test").mkdir()
    (base / "labeled_anomalies.csv").write_text(
        "\n".join(
            [
                "chan_id,spacecraft,anomaly_sequences,class,num_values",
                'A-1,SMAP,"[[1, 2]]",[point],10',
                'A-1,SMAP,"[[7, 8]]",[point],10',
                "A-2,SMAP,[],[point],10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for channel_id in ("A-1", "A-2"):
        np.save(base / "train" / f"{channel_id}.npy", np.zeros((20, 25)))
        np.save(base / "test" / f"{channel_id}.npy", np.zeros((10, 25)))

    bundle = load_real_benchmark("SMAP", tmp_path, 2, seed=42)

    assert bundle.details["source_entities"] == 2
    assert bundle.details["channel_ids"] == ["A-1", "A-2"]
    assert len(bundle.test_y) == 20
    assert int(bundle.test_y.sum()) == 4


def test_telemanom_splits_entities_into_requested_sensor_count(tmp_path):
    base = tmp_path / "telemanom"
    (base / "train").mkdir(parents=True)
    (base / "test").mkdir()
    (base / "labeled_anomalies.csv").write_text(
        "\n".join(
            [
                "chan_id,spacecraft,anomaly_sequences,class,num_values",
                'A-1,SMAP,"[[2, 4]]",[point],20',
                "A-2,SMAP,[],[point],20",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for channel_index, channel_id in enumerate(("A-1", "A-2")):
        train = np.full((20, 25), channel_index, dtype=np.float32)
        test = np.full((10, 25), channel_index, dtype=np.float32)
        np.save(base / "train" / f"{channel_id}.npy", train)
        np.save(base / "test" / f"{channel_id}.npy", test)

    bundle = load_real_benchmark("SMAP", tmp_path, 5)

    assert len(bundle.sensor_train) == 5
    assert all(len(samples) > 0 for samples in bundle.sensor_train.values())
    assert sum(map(len, bundle.sensor_train.values())) == 32
    assert bundle.details["source_entities"] == 2
    assert len(bundle.details["sensor_channel_ids"]) == 5
    assert set(bundle.details["sensor_channel_ids"]) == {"A-1", "A-2"}


def test_telemanom_dirichlet_partition_is_complete_and_alpha_sensitive(tmp_path):
    base = tmp_path / "telemanom"
    (base / "train").mkdir(parents=True)
    (base / "test").mkdir()
    rows = ["chan_id,spacecraft,anomaly_sequences,class,num_values"]
    for entity_index in range(6):
        channel_id = f"A-{entity_index + 1}"
        rows.append(f"{channel_id},SMAP,[],[point],120")
        train = np.full((120, 25), entity_index, dtype=np.float32)
        test = np.full((10, 25), entity_index, dtype=np.float32)
        np.save(base / "train" / f"{channel_id}.npy", train)
        np.save(base / "test" / f"{channel_id}.npy", test)
    (base / "labeled_anomalies.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )

    strong = load_real_benchmark(
        "SMAP", tmp_path, 20, seed=42, dirichlet_alpha=0.1
    )
    near_iid = load_real_benchmark(
        "SMAP", tmp_path, 20, seed=42, dirichlet_alpha=1.0e4
    )

    for bundle in (strong, near_iid):
        assert len(bundle.sensor_train) == 20
        assert all(len(samples) > 0 for samples in bundle.sensor_train.values())
        assert sum(map(len, bundle.sensor_train.values())) == 576
        assert bundle.details["partition_scheme"] == "source-entity-dirichlet"
    assert strong.partition_alpha == 0.1
    assert near_iid.partition_alpha == 1.0e4
    assert (
        strong.details["mean_normalised_client_entropy"]
        < near_iid.details["mean_normalised_client_entropy"]
    )


def test_hfl_smoke_run():
    net = copy.deepcopy(network_cfg)
    learn = copy.deepcopy(learning_cfg)
    energy = copy.deepcopy(energy_cfg)
    net.N_SENSORS = 6
    net.M_FOGS = 2
    net.MOBILITY_ENABLED = True
    learn.LOCAL_EPOCHS = 1
    learn.LOCAL_BATCH_SIZE = 8
    data = make_synthetic(
        6,
        samples_per_sensor=16,
        validation_samples=64,
        test_samples=96,
        seed=11,
    )
    simulator = AnomalyFLSimulator(
        data,
        net,
        acoustic_cfg,
        energy,
        learn,
        baseline="hfl-selective",
        seed=11,
        workers=2,
    )
    history = simulator.run(2)
    assert len(history) == 2
    assert history[-1]["participants"] > 0
    assert 0.0 <= history[-1]["f1"] <= 1.0
    assert history[-1]["e_round_comm_j"] > 0.0
    assert history[-1]["e_round_total_j"] == pytest.approx(
        history[-1]["e_round_comm_j"]
        + history[-1]["e_round_rx_j"]
        + history[-1]["e_round_compute_j"]
    )
    assert (
        history[-1]["e_cumulative_total_j"]
        > history[-1]["e_cumulative_comm_j"]
    )


def test_centralized_oracle_logs_raw_upload_energy_once():
    net = copy.deepcopy(network_cfg)
    learn = copy.deepcopy(learning_cfg)
    energy = copy.deepcopy(energy_cfg)
    net.N_SENSORS = 3
    net.M_FOGS = 1
    learn.LOCAL_EPOCHS = 1
    learn.LOCAL_BATCH_SIZE = 8
    data = make_synthetic(
        3,
        samples_per_sensor=8,
        validation_samples=16,
        test_samples=32,
        seed=23,
    )
    simulator = AnomalyFLSimulator(
        data,
        net,
        acoustic_cfg,
        energy,
        learn,
        baseline="centralized",
        seed=23,
        workers=1,
    )
    history = simulator.run(2)
    assert history[0]["e_round_comm_j"] > 0.0
    assert history[1]["e_round_comm_j"] == 0.0
    assert history[-1]["e_cumulative_comm_j"] == history[0]["e_round_comm_j"]
    assert history[1]["e_round_total_j"] == history[1]["e_round_compute_j"]
    assert (
        history[-1]["e_cumulative_total_j"]
        > history[-1]["e_cumulative_comm_j"]
    )
    assert simulator.metadata()["centralized_oracle_unconstrained"] is True
