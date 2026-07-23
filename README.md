# FL_Anomaly_Detection

Hierarchical Federated Anomaly Detection for the Internet of Underwater Things.

```text
surface gateway
       ↑
mobile fog aggregators (AUVs)
       ↑
stationary deep-water sensors
```

Sensor huấn luyện autoencoder không giám sát. Fog tổng hợp model update, có thể
trao đổi partial aggregate với fog lân cận rồi gửi model lên gateway. Link chỉ
tồn tại khi source level cần thiết để đạt target SNR không vượt `SLmax`.

Chi tiết phần kế thừa và phần thay đổi từ code FedKDL cũ nằm trong
[docs/FEDKDL_2D_OD_TO_ANOMALY_DIFF.md](docs/FEDKDL_2D_OD_TO_ANOMALY_DIFF.md).

## Cấu hình thực nghiệm SMAP/MSL

Ubuntu runner chỉ chạy hai benchmark:

- SMAP: input dimension `D=25`, 55 source telemetry channels.
- MSL: input dimension `D=55`, 27 source telemetry channels.

Mỗi dataset được ánh xạ thành cùng topology:

```text
N = 100 stationary sensors
M = 10 mobile fog/AUV aggregators
T = 30 federated rounds
seeds = 42, 43, 44
Dirichlet alpha = 0.1 and 10^4
```

Normal-training rows được phân phối cho đúng 100 sensor theo source-channel
Dirichlet: `α=0.1` là strongly non-IID, `α=10^4` là near-IID. Không sample nào
bị lặp hoặc bỏ; validation/test giữ nguyên. SMAP và MSL được train/evaluate độc
lập vì khác input dimension.

Mỗi dataset chạy tuần tự đúng thứ tự:

1. Centralised
2. FedAvg
3. FedProx
4. HFL-NoCoop
5. HFL-Selective
6. HFL-Nearest

Full matrix gồm `2 datasets × 2 α × 6 methods × 3 seeds = 72 runs`.

## Chạy trên Ubuntu server

### 1. Setup một lần

```bash
bash setup_ubuntu.sh
```

`setup_ubuntu.sh` chỉ làm phần chuẩn bị:

- Tạo `.venv`.
- Cài PyTorch CPU và toàn bộ `requirements.txt`.
- Tải và giải nén SMAP/MSL.
- Sinh và kiểm tra partition cho 2 dataset × 2 α × 3 seed.
- Ghi `datasets/partition_manifest.json` và setup log.

### 2. Chạy các kịch bản

Nên chạy smoke test trước. Quick mode vẫn kiểm tra đủ hai dataset, hai giá trị
α và sáu baseline, nhưng chỉ dùng seed 42 và hai round (`24 runs`):

```bash
QUICK=1 WORKERS=4 bash run_scenarios.sh
```

Chạy đủ 72 run, sau đó sinh Fig. 8 và Table IV:

```bash
WORKERS=8 bash run_scenarios.sh
```

Chạy lâu sau khi đóng SSH:

```bash
nohup env WORKERS=8 bash run_scenarios.sh > launcher.log 2>&1 &
```

`run_scenarios.sh` không cài package và không tải dataset. Nếu `.venv` hoặc
partition manifest chưa tồn tại, script dừng và yêu cầu chạy setup trước.

Các biến cấu hình:

- Cả hai file: `DATA_ROOT`, `OUTPUT_ROOT`, `VENV_DIR`.
- Setup: `PYTHON_BIN`, `TORCH_INDEX_URL`.
- Runner: `WORKERS` và `QUICK=1`.

## Log và kết quả

Raw log toàn phiên và result index có timestamp:

```text
results/setup_logs/
├── setup_<timestamp>.log
└── pip_freeze_<timestamp>.txt

results/runner_logs/
├── smap_msl_<timestamp>.log
├── smap_msl_<timestamp>_results.csv
└── smap_msl_<timestamp>_results.json
```

Mỗi dataset/baseline/seed lưu riêng:

```text
results/real/<dataset>/N_100_M_10/<baseline>/rho_0.05_alpha_<alpha>/seed_<seed>/
├── training.log
├── rounds.csv
├── metrics.json
└── summary.json
```

Sau full run:

```text
results/paper/
├── fig8_real_benchmarks.png
├── table_iv_real.csv
└── table_iv_real.md
```

`summary.json` và result index đều chứa communication energy, total modelled
energy (`E_tx + E_rx + E_comp`), latency, participation, F1 và PA-F1.

## Chạy một experiment thủ công

```bash
.venv/bin/python -u main.py \
  --scenario manual \
  --dataset SMAP \
  --baseline hfl-selective \
  --sensors 100 \
  --fogs 10 \
  --rounds 30 \
  --seed 42 \
  --workers 8
```

Các tên baseline hợp lệ:

```text
centralized
fedavg
fedprox
hfl-nocoop
hfl-selective
hfl-nearest
```

## Dataset

Runner chỉ tải package SMAP/MSL:

```bash
.venv/bin/python -m scripts.prepare_benchmarks --dataset nasa
```

Chi tiết source layout và cách chia 100 sensor nằm trong
[datasets/README.md](datasets/README.md).

## Kiểm thử

```bash
.venv/bin/python -m pytest -q
```
