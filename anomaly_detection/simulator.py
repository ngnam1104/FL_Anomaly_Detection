"""CPU-parallel simulator for flat and hierarchical federated anomaly detection."""

from __future__ import annotations

import copy
import logging
import multiprocessing as mp
import os
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import asdict
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from anomaly_detection.compression import (
    ErrorFeedbackTopK,
    flatten_state,
    unflatten_state,
)
from anomaly_detection.data import DataBundle
from anomaly_detection.model import Autoencoder, reconstruction_errors
from anomaly_detection.aggregation import (
    apply_weighted_deltas,
    blend_states,
    weighted_average,
)
from anomaly_detection.hfl_rules import select_cooperation
from anomaly_detection.metrics import anomaly_metrics, anomaly_threshold, joint_objective
from physics_models.energy import e_comp, e_rx, e_tx
from physics_models.latency import comm_delay, comp_delay, round_latency
from physics_models.communication import min_source_level, shannon_capacity
from physics_models.topology import (
    Topology3D,
    build_clusters,
    build_feasibility_graph,
    flat_feasible_sensors,
    nearest_feasible_association,
    topology_stats,
)


BASELINES = (
    "centralized",
    "fedavg",
    "fedprox",
    "hfl-nocoop",
    "hfl-selective",
    "hfl-nearest",
)


def _configure_local_worker(torch_threads: int) -> None:
    """Limit each worker to one BLAS/PyTorch thread to avoid oversubscription."""

    torch.set_num_threads(max(1, int(torch_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # It is already configured in a persistent worker.
        pass


def _train_sensor_worker(task: dict) -> dict:
    """Pickle-safe, CPU-only local training executed by a pool worker.

    Compression is deliberately left in the parent process because Top-K error
    feedback has mutable residual state for every sensor.
    """

    sensor_id = task["sensor_id"]
    global_state = task["global_state"]
    samples = task["samples"]
    model = Autoencoder(task["input_dim"], task["hidden_dims"])
    model.load_state_dict(global_state)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=task["local_lr"])
    generator = torch.Generator().manual_seed(task["train_seed"])
    loader = DataLoader(
        TensorDataset(samples),
        batch_size=task["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    global_parameters = {
        name: value.detach().clone() for name, value in global_state.items()
    }
    total_loss = 0.0
    batches = 0
    for _ in range(task["local_epochs"]):
        for (batch,) in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean(torch.sum((model(batch) - batch) ** 2, dim=1))
            if task["use_prox"]:
                proximal = sum(
                    torch.sum((parameter - global_parameters[name]) ** 2)
                    for name, parameter in model.named_parameters()
                )
                loss = loss + 0.5 * task["fedprox_mu"] * proximal
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite local loss: sensor={sensor_id} "
                    f"round={task['round_index']}"
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=task["max_grad_norm"],
                error_if_nonfinite=True,
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"Non-finite gradient: sensor={sensor_id} "
                    f"round={task['round_index']}"
                )
            optimizer.step()
            if not all(
                torch.isfinite(parameter).all() for parameter in model.parameters()
            ):
                raise FloatingPointError(
                    f"Non-finite local model: sensor={sensor_id} "
                    f"round={task['round_index']}"
                )
            total_loss += float(loss.detach())
            batches += 1
    local_state = model.state_dict()
    delta_state = {
        name: local_state[name].detach().cpu() - global_state[name].detach().cpu()
        for name in global_state
    }
    layer_ops = sum(
        module.in_features * module.out_features
        for module in model.modules()
        if isinstance(module, nn.Linear)
    )
    return {
        "sensor_id": sensor_id,
        "delta_state": delta_state,
        "samples": len(samples),
        "loss": total_loss / max(1, batches),
        "flops": len(samples) * task["local_epochs"] * layer_ops * 3.0,
    }


class AnomalyFLSimulator:
    def __init__(
        self,
        data: DataBundle,
        network_cfg,
        acoustic_cfg,
        energy_cfg,
        learning_cfg,
        *,
        baseline: str,
        seed: int = 42,
        workers: int = 4,
        parallel_backend: str = "auto",
        torch_threads: int = 1,
        logger: logging.Logger | None = None,
    ):
        if baseline not in BASELINES:
            raise ValueError(f"Unknown baseline {baseline!r}; choose from {BASELINES}")
        self.data = data
        self.net = copy.deepcopy(network_cfg)
        self.acoustic = copy.deepcopy(acoustic_cfg)
        self.energy = copy.deepcopy(energy_cfg)
        self.learning = copy.deepcopy(learning_cfg)
        self.baseline = baseline
        self.seed = int(seed)
        self.workers = max(1, int(workers))
        if parallel_backend not in {"auto", "process", "thread"}:
            raise ValueError("parallel_backend must be auto, process, or thread")
        self.parallel_backend = (
            "process"
            if parallel_backend == "auto" and os.name != "nt"
            else "thread"
            if parallel_backend == "auto"
            else parallel_backend
        )
        self.torch_threads = max(1, int(torch_threads))
        self.log = logger or logging.getLogger(__name__)
        self.net.N_SENSORS = len(data.sensor_train)
        self.learning.FEATURE_DIM = data.input_dim
        torch.manual_seed(self.seed)
        self.global_model = Autoencoder(data.input_dim, self.learning.HIDDEN_DIMS).cpu()
        self.topology = Topology3D(self.net, self.acoustic, self.seed)
        self.batteries = {
            sensor_id: float(self.energy.E_INIT) for sensor_id in data.sensor_train
        }
        flat, metadata = flatten_state(self.global_model.state_dict())
        self.state_metadata = metadata
        self.compressors = {
            sensor_id: ErrorFeedbackTopK(flat.numel(), self.learning.RHO_S)
            for sensor_id in data.sensor_train
        }
        self.model_bits = int(flat.numel() * 32)
        self.model_parameters = int(flat.numel())
        self.history: list[dict] = []
        self.centralized_cap_violations = 0

    @property
    def is_hierarchical(self) -> bool:
        return self.baseline.startswith("hfl-")

    def _train_sensor(
        self,
        sensor_id: int,
        global_state: Dict[str, torch.Tensor],
        round_index: int,
        *,
        compress_update: bool = True,
    ) -> dict:
        raw_result = _train_sensor_worker(
            self._training_task(sensor_id, global_state, round_index)
        )
        return self._finalise_local_update(raw_result, compress_update)

    def _training_task(
        self, sensor_id: int, global_state: Dict[str, torch.Tensor], round_index: int
    ) -> dict:
        return {
            "sensor_id": sensor_id,
            "global_state": global_state,
            "samples": self.data.sensor_train[sensor_id],
            "input_dim": self.data.input_dim,
            "hidden_dims": self.learning.HIDDEN_DIMS,
            "local_lr": self.learning.LOCAL_LR,
            "max_grad_norm": self.learning.MAX_GRAD_NORM,
            "local_epochs": self.learning.LOCAL_EPOCHS,
            "batch_size": self.learning.LOCAL_BATCH_SIZE,
            "fedprox_mu": self.learning.FEDPROX_MU,
            "use_prox": self.baseline == "fedprox",
            "round_index": round_index,
            "train_seed": self.seed * 1_000_003 + round_index * 10_007 + sensor_id,
        }

    def _finalise_local_update(self, raw_result: dict, compress_update: bool) -> dict:
        sensor_id = raw_result["sensor_id"]
        delta_state = raw_result["delta_state"]
        delta_vector, _ = flatten_state(delta_state)
        if not torch.isfinite(delta_vector).all():
            raise FloatingPointError(
                f"Non-finite update from sensor {sensor_id}"
            )
        if compress_update and self.learning.RHO_S < 1.0:
            payload = self.compressors[sensor_id].compress(delta_vector)
            reconstructed_delta = unflatten_state(payload.decompress(), self.state_metadata)
            payload_bits = payload.payload_bits
        elif compress_update:
            # The paper's rho=1 sensitivity point is an uncompressed FP32 upload.
            reconstructed_delta = delta_state
            payload_bits = self.model_bits
        else:
            reconstructed_delta = delta_state
            payload_bits = 0
        return {
            "sensor_id": sensor_id,
            "delta": reconstructed_delta,
            "samples": raw_result["samples"],
            "loss": raw_result["loss"],
            "payload_bits": payload_bits,
            "flops": raw_result["flops"],
        }

    def _local_training_pool(self):
        max_workers = self.workers
        if self.parallel_backend == "process":
            # Linux fork avoids re-importing the simulator for every worker and
            # keeps the CPU-only tensors independent from the parent process.
            kwargs = {"max_workers": max_workers, "initializer": _configure_local_worker,
                      "initargs": (self.torch_threads,)}
            if os.name != "nt":
                kwargs["mp_context"] = mp.get_context("fork")
            pool = ProcessPoolExecutor(**kwargs)
        else:
            pool = ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="sensor-train"
            )
        return pool

    def _parallel_local_training(
        self, participants: list[int], round_index: int, pool
    ) -> list[dict]:
        global_state = {
            name: value.detach().cpu().clone()
            for name, value in self.global_model.state_dict().items()
        }
        results = []
        participant_iter = iter(participants)
        futures = {}

        def submit_next() -> bool:
            try:
                sensor_id = next(participant_iter)
            except StopIteration:
                return False
            future = pool.submit(
                _train_sensor_worker,
                self._training_task(sensor_id, global_state, round_index),
            )
            futures[future] = sensor_id
            return True

        # ProcessPoolExecutor serialises tensor storages through multiprocessing
        # queues. Submitting every sensor at once can therefore exhaust Linux file
        # descriptors even though only ``workers`` tasks execute concurrently.
        # Keep the queue bounded while continuously replenishing completed work.
        for _ in range(min(self.workers, len(participants))):
            submit_next()

        try:
            while futures:
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    futures.pop(future)
                    result = self._finalise_local_update(
                        future.result(), compress_update=True
                    )
                    results.append(result)
                    self.log.debug(
                        "round=%d sensor=%d samples=%d loss=%.6f payload_bits=%d",
                        round_index,
                        result["sensor_id"],
                        result["samples"],
                        result["loss"],
                        result["payload_bits"],
                    )
                    submit_next()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        return sorted(results, key=lambda item: item["sensor_id"])

    def _evaluate(self) -> dict:
        validation_errors = reconstruction_errors(
            self.global_model, self.data.validation_normal
        ).numpy()
        if not np.isfinite(validation_errors).all():
            raise FloatingPointError("Non-finite validation reconstruction errors")
        threshold = anomaly_threshold(
            validation_errors, self.learning.ANOMALY_PERCENTILE
        )
        test_errors = reconstruction_errors(self.global_model, self.data.test_x).numpy()
        if not np.isfinite(test_errors).all():
            raise FloatingPointError("Non-finite test reconstruction errors")
        metrics = anomaly_metrics(self.data.test_y, test_errors, threshold)
        metrics["threshold"] = threshold
        metrics["test_reconstruction_loss"] = float(np.mean(test_errors))
        return metrics

    def _link_energy(self, bits: int, link) -> tuple[float, float]:
        tx = e_tx(
            bits,
            link.R_bps,
            link.SL_min,
            self.energy.ETA_EA,
            self.energy.P_C_TX,
            self.energy.RHO_WATER,
            self.acoustic.SOUND_SPEED,
        )
        return tx, e_rx(bits, link.R_bps, self.energy.P_C_RX)

    def run(self, rounds: int) -> list[dict]:
        if self.baseline == "centralized":
            return self._run_centralized(rounds)
        return self._run_federated(rounds)

    def _run_federated(self, rounds: int) -> list[dict]:
        pool = self._local_training_pool()
        try:
            return self._run_federated_with_pool(rounds, pool)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    def _run_federated_with_pool(self, rounds: int, pool) -> list[dict]:
        cumulative_comm_energy = 0.0
        cumulative_total_energy = 0.0
        cumulative_latency = 0.0
        for round_index in range(1, rounds + 1):
            mobility = {"avg_move_m": 0.0, "max_move_m": 0.0, "avg_speed_mps": 0.0}
            if (
                round_index > 1
                and self.net.MOBILITY_ENABLED
                and self.topology.M > 0
            ):
                mobility = self.topology.step_mobile_fogs()
            graph = build_feasibility_graph(self.topology, self.acoustic)
            if self.is_hierarchical:
                association = nearest_feasible_association(self.topology, graph)
                clusters = build_clusters(association, self.topology.M)
                participants = sorted(association)
            else:
                participants = flat_feasible_sensors(self.topology, graph)
                association = {sensor_id: -1 for sensor_id in participants}
                clusters = {-1: participants}
            participants = [
                sensor_id
                for sensor_id in participants
                if self.batteries[sensor_id] > self.energy.E_MIN
                and len(self.data.sensor_train[sensor_id]) > 0
            ]
            if not participants:
                raise RuntimeError(f"No feasible participants in round {round_index}")
            self.log.debug(
                "round=%d feasible_edges=%d association=%s clusters=%s mobility=%s",
                round_index,
                len(graph),
                association,
                clusters,
                mobility,
            )
            local_results = self._parallel_local_training(participants, round_index, pool)
            by_sensor = {item["sensor_id"]: item for item in local_results}
            global_before = {
                name: value.detach().cpu().clone()
                for name, value in self.global_model.state_dict().items()
            }

            e_s2f = e_f2f = e_f2g = e_rx_total = 0.0
            link_delays: list[float] = []
            max_compute_delay = 0.0
            e_compute = 0.0
            for result in local_results:
                sensor_id = result["sensor_id"]
                target_kind = "fog" if self.is_hierarchical else "gateway"
                target_id = association[sensor_id] if self.is_hierarchical else 0
                link = graph[("sensor", sensor_id, target_kind, target_id)]
                tx, rx = self._link_energy(result["payload_bits"], link)
                e_s2f += tx
                e_rx_total += rx
                link_delays.append(
                    comm_delay(
                        result["payload_bits"],
                        link.R_bps,
                        link.distance,
                        self.acoustic.SOUND_SPEED,
                    )
                )
                comp_j = e_comp(result["flops"], self.energy.EPSILON_OP)
                e_compute += comp_j
                self.batteries[sensor_id] -= tx + comp_j
                max_compute_delay = max(
                    max_compute_delay,
                    comp_delay(
                        result["flops"],
                        self.energy.F_CPU,
                        self.energy.N_CORES,
                        self.energy.FLOPS_PER_CYCLE,
                    ),
                )

            cooperation: Dict[int, int] = {}
            if self.is_hierarchical:
                fog_states = {}
                cluster_weights = {}
                for fog_id, members in clusters.items():
                    active_members = [member for member in members if member in by_sensor]
                    if not active_members:
                        continue
                    results = [by_sensor[member] for member in active_members]
                    fog_states[fog_id] = apply_weighted_deltas(
                        global_before,
                        [result["delta"] for result in results],
                        [result["samples"] for result in results],
                    )
                    cluster_weights[fog_id] = sum(result["samples"] for result in results)
                rule = self.baseline.removeprefix("hfl-")
                cooperation = select_cooperation(
                    rule,
                    {fog_id: len([m for m in members if m in by_sensor]) for fog_id, members in clusters.items()},
                    graph,
                    self.learning.COOP_THRESHOLD_MULTIPLIER,
                )
                mixed_states = {fog_id: state for fog_id, state in fog_states.items()}
                neighbour_weight = (
                    self.learning.COOP_NEIGHBOR_WEIGHT_NEAREST
                    if rule == "nearest"
                    else self.learning.COOP_NEIGHBOR_WEIGHT_SELECTIVE
                )
                for receiver, donor in cooperation.items():
                    if receiver not in fog_states or donor not in fog_states:
                        continue
                    mixed_states[receiver] = blend_states(
                        fog_states[receiver], fog_states[donor], neighbour_weight
                    )
                    link = graph[("fog", donor, "fog", receiver)]
                    tx, rx = self._link_energy(self.model_bits, link)
                    e_f2f += tx
                    e_rx_total += rx
                    link_delays.append(
                        comm_delay(
                            self.model_bits,
                            link.R_bps,
                            link.distance,
                            self.acoustic.SOUND_SPEED,
                        )
                    )
                for fog_id in mixed_states:
                    link = graph[("fog", fog_id, "gateway", 0)]
                    tx, rx = self._link_energy(self.model_bits, link)
                    e_f2g += tx
                    e_rx_total += rx
                    link_delays.append(
                        comm_delay(
                            self.model_bits,
                            link.R_bps,
                            link.distance,
                            self.acoustic.SOUND_SPEED,
                        )
                    )
                new_state = weighted_average(
                    [mixed_states[fog_id] for fog_id in sorted(mixed_states)],
                    [cluster_weights[fog_id] for fog_id in sorted(mixed_states)],
                )
            else:
                new_state = apply_weighted_deltas(
                    global_before,
                    [result["delta"] for result in local_results],
                    [result["samples"] for result in local_results],
                )
            if not all(torch.isfinite(value).all() for value in new_state.values()):
                raise FloatingPointError(
                    f"Non-finite global state at round {round_index}"
                )
            self.global_model.load_state_dict(new_state)
            eval_metrics = self._evaluate()
            comm_energy = e_s2f + e_f2f + e_f2g
            total_energy = comm_energy + e_rx_total + e_compute
            latency = round_latency(link_delays, max_compute_delay)
            cumulative_comm_energy += comm_energy
            cumulative_total_energy += total_energy
            cumulative_latency += latency
            train_loss = float(
                np.average(
                    [result["loss"] for result in local_results],
                    weights=[result["samples"] for result in local_results],
                )
            )
            record = {
                "round": round_index,
                "train_loss": train_loss,
                **eval_metrics,
                "participants": len(participants),
                "participation": len(participants) / self.topology.N,
                "e_round_comm_j": comm_energy,
                "e_round_rx_j": e_rx_total,
                "e_round_compute_j": e_compute,
                "e_round_total_j": total_energy,
                "e_sensor_upload_j": e_s2f,
                "e_s2f_j": e_s2f if self.is_hierarchical else 0.0,
                "e_s2g_j": 0.0 if self.is_hierarchical else e_s2f,
                "e_f2f_j": e_f2f,
                "e_f2g_j": e_f2g,
                "e_cumulative_comm_j": cumulative_comm_energy,
                "e_cumulative_total_j": cumulative_total_energy,
                "latency_round_s": latency,
                "latency_cumulative_s": cumulative_latency,
                "joint_objective": joint_objective(
                    train_loss,
                    total_energy,
                    latency,
                    self.learning.LAMBDA_E,
                    self.learning.LAMBDA_TAU,
                ),
                "cooperation_links": len(cooperation),
                "payload_sensor_mean_bits": float(
                    np.mean([result["payload_bits"] for result in local_results])
                ),
                **mobility,
            }
            self.history.append(record)
            self.log.debug(
                "round=%d energy_breakdown s2f=%.6f f2f=%.6f f2g=%.6f "
                "rx=%.6f compute=%.6f total=%.6f threshold=%.8f",
                round_index,
                e_s2f,
                e_f2f,
                e_f2g,
                e_rx_total,
                e_compute,
                total_energy,
                record["threshold"],
            )
            self.log.info(
                "round=%d/%d method=%s participants=%d/%d loss=%.6f "
                "f1=%.4f pa_f1=%.4f energy_total=%.4fJ "
                "energy_comm=%.4fJ latency=%.4fs coop=%d",
                round_index,
                rounds,
                self.baseline,
                len(participants),
                self.topology.N,
                train_loss,
                record["f1"],
                record["pa_f1"],
                total_energy,
                comm_energy,
                latency,
                len(cooperation),
            )
        return self.history

    def _run_centralized(self, rounds: int) -> list[dict]:
        pooled = torch.cat(list(self.data.sensor_train.values()), dim=0)
        original = self.data.sensor_train
        raw_bits_by_sensor = self.data.raw_bits_by_sensor()
        raw_energy = 0.0
        raw_rx = 0.0
        raw_delays = []
        rate = shannon_capacity(
            self.acoustic.BANDWIDTH, self.acoustic.TARGET_SNR
        )
        self.centralized_cap_violations = 0
        for sensor_id, bits in raw_bits_by_sensor.items():
            distance = float(
                np.linalg.norm(
                    self.topology.sensor_positions[sensor_id]
                    - self.topology.gateway_position
                )
            )
            required_sl = min_source_level(
                distance,
                self.acoustic.CARRIER_FREQ,
                self.acoustic.BANDWIDTH,
                self.acoustic.TARGET_SNR,
                self.acoustic.IL_LOSS,
                self.acoustic.SPREADING_FACTOR,
                self.acoustic.WIND_SPEED,
                self.acoustic.SHIPPING_FACTOR,
            )
            if required_sl > self.acoustic.SL_MAX:
                self.centralized_cap_violations += 1
            raw_energy += e_tx(
                bits,
                rate,
                required_sl,
                self.energy.ETA_EA,
                self.energy.P_C_TX,
                self.energy.RHO_WATER,
                self.acoustic.SOUND_SPEED,
            )
            raw_rx += e_rx(bits, rate, self.energy.P_C_RX)
            raw_delays.append(
                comm_delay(
                    bits, rate, distance, self.acoustic.SOUND_SPEED
                )
            )
        self.data.sensor_train = {0: pooled}
        self.compressors = {
            0: ErrorFeedbackTopK(self.model_parameters, self.learning.RHO_S)
        }
        cumulative_comm_energy = 0.0
        cumulative_total_energy = 0.0
        cumulative_latency = 0.0
        for round_index in range(1, rounds + 1):
            state = {
                name: value.detach().cpu().clone()
                for name, value in self.global_model.state_dict().items()
            }
            result = self._train_sensor(
                0, state, round_index, compress_update=False
            )
            new_state = apply_weighted_deltas(
                state, [result["delta"]], [result["samples"]]
            )
            if not all(torch.isfinite(value).all() for value in new_state.values()):
                raise FloatingPointError(
                    f"Non-finite centralized state at round {round_index}"
                )
            self.global_model.load_state_dict(new_state)
            metrics = self._evaluate()
            compute_energy = e_comp(result["flops"], self.energy.EPSILON_OP)
            compute_latency = comp_delay(
                result["flops"],
                self.energy.F_CPU,
                self.energy.N_CORES,
                self.energy.FLOPS_PER_CYCLE,
            )
            round_comm_energy = raw_energy if round_index == 1 else 0.0
            round_rx_energy = raw_rx if round_index == 1 else 0.0
            round_total_energy = (
                round_comm_energy + round_rx_energy + compute_energy
            )
            latency = round_latency(
                raw_delays if round_index == 1 else [], compute_latency
            )
            cumulative_comm_energy += round_comm_energy
            cumulative_total_energy += round_total_energy
            cumulative_latency += latency
            record = {
                "round": round_index,
                "train_loss": result["loss"],
                **metrics,
                "participants": self.topology.N,
                "participation": 1.0,
                "e_round_comm_j": round_comm_energy,
                "e_round_rx_j": round_rx_energy,
                "e_round_compute_j": compute_energy,
                "e_round_total_j": round_total_energy,
                "e_sensor_upload_j": round_comm_energy,
                "e_s2f_j": 0.0,
                "e_s2g_j": round_comm_energy,
                "e_f2f_j": 0.0,
                "e_f2g_j": 0.0,
                "e_cumulative_comm_j": cumulative_comm_energy,
                "e_cumulative_total_j": cumulative_total_energy,
                "latency_round_s": latency,
                "latency_cumulative_s": cumulative_latency,
                "joint_objective": joint_objective(
                    result["loss"],
                    round_total_energy,
                    latency,
                    self.learning.LAMBDA_E,
                    self.learning.LAMBDA_TAU,
                ),
                "cooperation_links": 0,
                "payload_sensor_mean_bits": 0.0,
                "avg_move_m": 0.0,
                "max_move_m": 0.0,
                "avg_speed_mps": 0.0,
            }
            self.history.append(record)
            self.log.info(
                "round=%d/%d method=centralized loss=%.6f f1=%.4f "
                "pa_f1=%.4f energy_total=%.4fJ energy_comm=%.4fJ",
                round_index,
                rounds,
                result["loss"],
                record["f1"],
                record["pa_f1"],
                round_total_energy,
                round_comm_energy,
            )
        self.data.sensor_train = original
        return self.history

    def metadata(self) -> dict:
        graph = build_feasibility_graph(self.topology, self.acoustic)
        return {
            "dataset": self.data.name,
            "baseline": self.baseline,
            "seed": self.seed,
            "workers": self.workers,
            "parallel_backend": self.parallel_backend,
            "torch_threads_per_worker": self.torch_threads,
            "input_dim": self.data.input_dim,
            "model_parameters": self.model_parameters,
            "partition_alpha": self.data.partition_alpha,
            "data_details": self.data.details,
            "centralized_oracle_unconstrained": self.baseline == "centralized",
            "centralized_source_cap_violations": self.centralized_cap_violations,
            "topology": topology_stats(self.topology, graph),
            "network_config": asdict(self.net),
            "acoustic_config": asdict(self.acoustic),
            "energy_config": asdict(self.energy),
            "learning_config": asdict(self.learning),
        }
