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
```

Training sequence của các source channel được chia thành đúng 100 đoạn liên
tục, không chồng lặp và không nhân bản sample. Mỗi sensor chỉ chứa shard của một
source channel. SMAP và MSL được train/evaluate độc lập vì khác input dimension.

Mỗi dataset chạy tuần tự đúng thứ tự:

1. Centralised
2. FedAvg
3. FedProx
4. HFL-NoCoop
5. HFL-Selective
6. HFL-Nearest

Full matrix gồm `2 datasets × 6 methods × 3 seeds = 36 runs`.

## Chạy trên Ubuntu server

Một lệnh sẽ tạo virtual environment, cài PyTorch CPU và dependency, tải/kiểm
tra SMAP/MSL, chạy đủ 36 run, vẽ Fig. 8 và sinh Table IV:

```bash
bash run_ubuntu.sh all
```

Nên chạy smoke test trước. Quick mode vẫn kiểm tra đủ hai dataset và sáu
baseline, nhưng chỉ dùng seed 42 và hai round:

```bash
QUICK=1 WORKERS=4 bash run_ubuntu.sh run
```

Chạy lâu sau khi đóng SSH:

```bash
nohup env WORKERS=8 bash run_ubuntu.sh all > launcher.log 2>&1 &
```

Các action:

```bash
bash run_ubuntu.sh install
bash run_ubuntu.sh prepare-data
bash run_ubuntu.sh run
PREPARE_DATA=0 WORKERS=8 bash run_ubuntu.sh run
```

Các biến cấu hình runner:

- `WORKERS`: số sensor được local-train đồng thời, mặc định tối đa 8.
- `QUICK=1`: một seed, hai round, nhưng vẫn chạy đủ sáu baseline.
- `PREPARE_DATA=0`: bỏ qua download/extract khi data đã sẵn sàng.
- `DATA_ROOT`, `OUTPUT_ROOT`, `VENV_DIR`, `PYTHON_BIN`: ghi đè đường dẫn/runtime.

## Log và kết quả

Raw log toàn phiên và result index có timestamp:

```text
results/runner_logs/
├── <action>_smap_msl_<timestamp>.log
├── <action>_smap_msl_<timestamp>_results.csv
├── <action>_smap_msl_<timestamp>_results.json
└── pip_freeze_<timestamp>.txt
```

Mỗi dataset/baseline/seed lưu riêng:

```text
results/real/<dataset>/N_100_M_10/<baseline>/rho_0.05_alpha_na/seed_<seed>/
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
