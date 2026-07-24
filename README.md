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

## Cấu hình thực nghiệm của paper

Runner tái tạo toàn bộ Fig. 4-8 và Table II-IV:

- synthetic: scalability, compression và non-IID sensitivity.
- SMD: `D=38`, 10 source machines.
- SMAP: input dimension `D=25`; PDF reports 55 rows, while the bundled
  metadata resolves to 54 unique channels after merging duplicate `P-2`.
- MSL: input dimension `D=55`, 27 source telemetry channels.

Ba benchmark thật được ánh xạ thành cùng topology:

```text
N = 100 stationary sensors
M = 10 mobile fog/AUV aggregators
T = 30 federated rounds
seeds = 42, 43, 44
```

Các source machine/channel được chia thành các shard liên tục, không lặp hoặc
bỏ sample, để tạo đúng 100 sensor; validation/test giữ nguyên. Thí nghiệm
Dirichlet `α=0.1` và `α=10^4` của Fig. 7 dùng dữ liệu synthetic như trong paper.

Mỗi dataset chạy tuần tự đúng thứ tự:

1. Centralised
2. FedAvg
3. FedProx
4. HFL-NoCoop
5. HFL-Selective
6. HFL-Nearest

Full matrix gồm 162 run:

- convergence extras cho Fig. 4: `2 N × 2 methods × 3 seeds = 12`;
- scalability: `4 N × 4 methods × 3 seeds = 48`;
- compression: `4 methods × 2 ρ × 3 seeds = 24`;
- non-IID: `4 methods × 2 α × 3 seeds = 24`;
- real: `3 datasets × 6 methods × 3 seeds = 54`.

## Chạy trên Ubuntu server

### 1. Setup một lần

```bash
bash setup_ubuntu.sh
```

Nếu Ubuntu báo thiếu `ensurepip`, cài gói venv đúng phiên bản Python rồi chạy
lại. Ví dụ log của Python 3.13 cần:

```bash
sudo apt-get update && sudo apt-get install -y python3.13-venv
bash setup_ubuntu.sh
```

Hoặc để setup script tự gọi `sudo apt-get`:

```bash
INSTALL_SYSTEM_VENV=1 bash setup_ubuntu.sh
```

`setup_ubuntu.sh` chỉ làm phần chuẩn bị:

- Tạo `.venv`.
- Cài PyTorch CPU và toàn bộ `requirements.txt`.
- Tải và giải nén SMD, SMAP và MSL.
- Sinh và kiểm tra real-data partition cho 3 dataset × 3 seed.
- Ghi `datasets/partition_manifest.json` và setup log.

### 2. Chạy các kịch bản

Nên chạy smoke test trước. Quick mode đi qua cả năm scenario, ba real dataset
và sáu baseline, nhưng chỉ dùng seed 42, hai round và ma trận synthetic rút gọn
(`25 runs`):

```bash
QUICK=1 WORKERS=4 bash run_scenarios.sh
```

Chạy đủ 162 run, kiểm tra tính đầy đủ/hữu hạn của kết quả, sau đó sinh toàn bộ
Fig. 4-8 và Table II-IV:

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
├── paper_all_<timestamp>.log
├── paper_all_<timestamp>_results.csv
└── paper_all_<timestamp>_results.json
```

Mỗi dataset/baseline/seed lưu riêng:

```text
results/<scenario>/<dataset>/N_<N>_M_<M>/<baseline>/rho_<rho>_alpha_<alpha>/seed_<seed>/
├── training.log
├── rounds.csv
├── metrics.json
└── summary.json
```

Sau full run:

```text
results/paper/
├── fig4_convergence.png
├── fig5_scalability.png
├── fig6_engineering.png
├── fig7_noniid.png
├── fig8_real_benchmarks.png
├── table_ii_parameters.{csv,md}
├── table_iii_scalability.{csv,md}
└── table_iv_real.{csv,md}
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

Chuẩn bị cả ba benchmark thật:

```bash
.venv/bin/python -m scripts.prepare_benchmarks --dataset all
```

Chi tiết source layout và cách chia 100 sensor nằm trong
[datasets/README.md](datasets/README.md).

## Kiểm thử

```bash
.venv/bin/python -m pytest -q
```
