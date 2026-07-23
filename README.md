# FL_Anomaly_Detection

Hierarchical Federated Anomaly Detection for the Internet of Underwater Things.

Mô phỏng ba tầng theo Omeke et al. (2026):

```text
surface gateway
       ↑
mobile fog aggregators (AUVs)
       ↑
stationary deep-water sensors
```

Sensor huấn luyện autoencoder không giám sát. Fog tổng hợp update, có thể trao
đổi partial aggregate với fog lân cận, rồi gửi model tới gateway. Link chỉ tồn
tại khi source level cần thiết để đạt target SNR không vượt `SLmax`.

Đọc [tài liệu diff với FedKDL 2D OD](docs/FEDKDL_2D_OD_TO_ANOMALY_DIFF.md)
để biết chính xác phần physics nào được giữ, phần nào thay đổi và lý do.

## Thành phần

- Thorp transmission loss, Wenz ambient noise và target-SNR power control.
- Energy báo cáo gồm transmit + receive + local-training compute; communication
  energy theo Eq. (20) vẫn được log riêng.
- Sensor tĩnh; fog/AUV di động Gauss-Markov giữa các round.
- Autoencoder `D→16→8→16→D`; `D=32` có đúng 1.352 tham số.
- Top-K `rho_s=0.05`, INT8 và error feedback cho sensor upload.
- Fog-to-fog và fog-to-gateway dùng model FP32.
- Centralized, FedAvg, FedProx, HFL-NoCoop, HFL-Selective, HFL-Nearest.
- Threshold percentile 99 trên validation normal-only.
- Point-wise F1 cho synthetic và PA-F1 cho real benchmarks.
- Local training song song bằng `ThreadPoolExecutor` trên CPU.
- Log chi tiết, round CSV, metrics JSON và summary JSON.

Thông số Table II nằm trong [config/settings.py](config/settings.py).

## Cài đặt Windows

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Chạy trên Ubuntu server

Một script duy nhất có thể tạo virtualenv, cài PyTorch CPU và các dependency,
tải/kiểm tra benchmark, sau đó chạy từng baseline:

```bash
bash run_ubuntu.sh all
```

Full `all` gồm 150 run (3 seed, synthetic 20 round, real 30 round), nên nên chạy
trong `tmux` hoặc dùng:

```bash
nohup env WORKERS=8 bash run_ubuntu.sh all > launcher.log 2>&1 &
```

Dry-run trước khi chạy dài:

```bash
QUICK=1 WORKERS=4 bash run_ubuntu.sh run all
```

Chạy riêng một suite:

```bash
WORKERS=8 bash run_ubuntu.sh run scalability
WORKERS=8 bash run_ubuntu.sh run compression
WORKERS=8 bash run_ubuntu.sh run noniid
WORKERS=8 bash run_ubuntu.sh run real
```

Các action khác:

```bash
bash run_ubuntu.sh install
bash run_ubuntu.sh prepare-data
PREPARE_DATA=0 bash run_ubuntu.sh run real
```

Raw log toàn phiên và bảng tổng hợp có timestamp (`*_results.csv`,
`*_results.json`) nằm trong `results/runner_logs/`. Mỗi baseline/seed có thêm
`training.log`, `rounds.csv`, `metrics.json` và `summary.json` trong thư mục
experiment tương ứng. Khi hoàn tất full suite `all`, script tự sinh Figure
4–8 và Table II–IV vào `results/paper/`.

## Chạy một experiment

```powershell
python -u main.py --scenario manual --dataset synthetic `
  --baseline hfl-selective --sensors 50 --fogs 5 `
  --rounds 20 --seed 42 --workers 8
```

Artifact:

```text
results/manual/synthetic/N_50_M_5/hfl-selective/rho_0.05_alpha_1.0/seed_42/
├── training.log
├── rounds.csv
├── metrics.json
└── summary.json
```

`--workers` là số sensor train đồng thời. PyTorch mặc định dùng một intra-op
thread cho mỗi worker để tránh CPU oversubscription.

## Chạy các scenario của paper

```powershell
# Fig. 4, Fig. 5, Fig. 6(a), Table III
.\run_windows.ps1 -Suite scalability -Workers 8

# Fig. 6(b): rho=0.05 so với full FP32
.\run_windows.ps1 -Suite compression -Workers 8

# Fig. 7: Dirichlet alpha=0.1 và alpha=10^4
.\run_windows.ps1 -Suite noniid -Workers 8

# Fig. 8 và Table IV
.\run_windows.ps1 -Suite real -Workers 8

# Tất cả
.\run_windows.ps1 -Suite all -Workers 8
```

Thêm `-Quick` để dùng một seed, hai round và tập synthetic nhỏ. Suite `real`
vẫn cần SMD/SMAP/MSL ngay cả khi bật `-Quick`.

## Sinh đúng hình và bảng Section VI

Sau khi chạy đủ scenario:

```powershell
python -m scripts.paper.plot_all --results results --output results/paper
```

Có thể chạy riêng:

```powershell
python -m scripts.paper.fig4_convergence
python -m scripts.paper.fig5_scalability
python -m scripts.paper.fig6_engineering
python -m scripts.paper.fig7_noniid
python -m scripts.paper.fig8_real
python -m scripts.paper.tables
```

Kết quả gồm Fig. 4–8 và Table II–IV dưới `results/paper/`.

## Chuẩn bị SMD, SMAP và MSL

Tải và giải nén đúng cấu trúc entity/client:

```powershell
python -m scripts.prepare_benchmarks
```

- SMD dùng 10 trong 28 machine, mỗi machine là một client `D=38`.
- SMAP dùng 55 telemetry channel/client, `D=25`.
- MSL dùng 27 telemetry channel/client, `D=55`.

Chi tiết nguồn và layout nằm tại [datasets/README.md](datasets/README.md).

Loader vẫn hỗ trợ layout processed tương thích ngược:

```text
datasets/
├── SMD_train.npy
├── SMD_test.npy
├── SMD_test_label.npy
├── SMAP_train.npy
├── SMAP_test.npy
├── SMAP_test_label.npy
├── MSL_train.npy
├── MSL_test.npy
└── MSL_test_label.npy
```

hoặc `datasets/<NAME>/{train.npy,test.npy,labels.npy}`. Raw SMD:

```text
datasets/SMD/{train,test,test_label}/*.txt
```

Train được tách 90/10 để lấy validation normal-only. Chuẩn hóa chỉ dùng thống
kê train; test label không được dùng để chọn threshold.

## Kiểm thử

```powershell
pytest -q
```
