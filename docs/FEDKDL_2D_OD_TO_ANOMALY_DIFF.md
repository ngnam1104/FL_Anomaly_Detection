# FedKDL 2D Object Detection → FL_Anomaly_Detection

Tài liệu này ghi lại phạm vi chuyển đổi từ codebase FedKDL cũ sang simulator
hierarchical federated anomaly detection. Mục tiêu là giữ kiến trúc ba tầng và
mô hình kênh âm dưới nước có thể tái sử dụng, đồng thời thay đúng vai trò vật
lý, learning task, aggregation method và ma trận thực nghiệm của paper anomaly.

## 1. Thay đổi kiến trúc hệ thống

| FedKDL 2D OD cũ | FL_Anomaly_Detection mới |
| --- | --- |
| Gateway ở mặt nước | Gateway ở mặt nước, giữ vai trò global coordinator |
| Fog/relay tĩnh ở tầng giữa | Fog/AUV ở tầng giữa, di chuyển giữa các FL round |
| AUV ở tầng sâu, mang dữ liệu ảnh | Sensor tĩnh ở tầng sâu, mang time-series cục bộ |
| AUV → relay → gateway | Sensor → fog(AUV) → gateway |
| Object detection với YOLO/LoRA/KD | Unsupervised anomaly detection với autoencoder |

Vị trí vẫn là vector 3D. Sensor và fog được lấy mẫu đều trong vùng ngang và
trong khoảng độ sâu của tầng tương ứng. Sensor không di chuyển. Fog dùng
Gauss–Markov mobility và được xem là quasi-static trong một round.

Topology không còn được hiểu là một graph cố định. Sau khi fog di chuyển, code
tính lại khoảng cách 3D, source level tối thiểu và tập cạnh khả thi theo
`SL_min <= SL_max`.

## 2. Mô hình vật lý: phần giữ nguyên và phần thay đổi

### 2.1 Các phương trình được giữ nguyên

Các hàm sau giữ cùng công thức số học như FedKDL:

- Hệ số hấp thụ Thorp `alpha(f)`.
- Transmission loss:
  `TL(d,f) = 10 k log10(d) + alpha(f)d/1000`.
- Bốn thành phần Wenz: turbulence, shipping, wind và thermal.
- Cộng noise trong miền tuyến tính rồi tích phân theo bandwidth.
- Passive-sonar SNR:
  `SNR = SL - TL - NL - IL`.
- Source level tối thiểu:
  `SL_min = gamma_tgt + TL + NL + IL`.
- Capped-source-level feasibility:
  một cạnh tồn tại khi `SL_min <= SL_max`.
- Shannon rate tại target SNR.
- Chuyển source level sang acoustic power.
- Transmit energy và receive-circuit energy.

`tests/test_core.py::test_acoustic_equations_match_fedkdl_regression_values`
khóa các giá trị regression để phát hiện thay đổi ngoài ý muốn ở nhóm công
thức này.

Các chỉnh sửa trong `physics_models/communication.py` chủ yếu là:

- Đổi tên tham số cho khớp paper anomaly.
- Thêm type hints và kiểm tra input không hợp lệ.
- Sửa docstring/số phương trình theo tài liệu mới.
- Trả về scalar `float` nhất quán.

Những chỉnh sửa đó không làm đổi công thức Thorp–Wenz–SNR.

### 2.2 Các thay đổi vật lý thực sự

1. **Vai trò node và mobility**

   Mobility chuyển từ tầng dữ liệu/AUV cũ sang fog/AUV tầng giữa. Sensor tầng
   sâu đứng yên. Vì vậy link sensor–fog và fog–fog thay đổi giữa các round.

2. **Payload**

   Payload ảnh, LoRA và knowledge-distillation state được bỏ. Sensor upload
   model-update autoencoder đã Top-K `rho_s=0.05` và quantise INT8. Fog–fog và
   fog–gateway trao đổi FP32.

3. **Đường truyền năng lượng**

   Năng lượng truyền được phân rã thành sensor–fog, sensor–gateway fallback,
   fog–fog và fog–gateway. Receive energy được log riêng.

4. **Computation energy**

   FedKDL cũ dùng:

   ```text
   Ecomp = epsilon_op × Phi × f_cpu²
   ```

   Code mới dùng mô hình của paper anomaly:

   ```text
   Ecomp = epsilon_op × Phi
   ```

   `epsilon_op = 2.8e-10 J/FLOP` là simulator option được hiệu chỉnh từ budget
   thiết bị 10 W và throughput CPU cấu hình; nó không được trình bày như một
   hàng của Table II.

5. **Tổng năng lượng**

   Simulator log cả:

   ```text
   E_comm = E_tx
   E_total = E_tx + E_rx + E_comp
   ```

   Battery và joint objective dùng `E_total` theo yêu cầu triển khai. Thành
   phần communication-only vẫn được giữ để đối chiếu các phương trình năng
   lượng truyền thông của paper.

Mobility propulsion energy của FedKDL cũ bị loại vì paper anomaly không đưa nó
vào objective và không cung cấp đủ tham số vehicle/thruster trong Table II.

## 3. Learning task

| Thành phần | Cũ | Mới |
| --- | --- | --- |
| Input | Ảnh URPC/COCO | Vector time-series đa biến |
| Model | YOLO + LoRA/KD | AE đối xứng `D→16→8→16→D` |
| `D=32` | Không áp dụng | 1.352 trainable parameters |
| Local loss | Detection/KD losses | Mean squared reconstruction error |
| Threshold | Detection confidence | Percentile 99 trên validation normal-only |
| Metric | mAP/precision/recall | F1 và point-adjusted F1 |

Validation được tách từ normal training data. Thống kê chuẩn hóa chỉ được fit
trên train. Test labels không tham gia chọn threshold.

## 4. Federated methods

Code mới giữ các baseline cần cho paper:

- `centralized`: oracle upper bound, không bị acoustic topology giới hạn.
- `fedavg`: flat feasible-client FedAvg.
- `fedprox`: flat FL với proximal local objective.
- `hfl-nocoop`: sensor → fog aggregation → gateway.
- `hfl-nearest`: mỗi fog trao đổi với fog khả thi gần nhất.
- `hfl-selective`: chỉ trao đổi khi cooperative score đạt ngưỡng.

`federated_core` đã được gộp vào package `anomaly_detection`; logic aggregation,
compression, HFL rules, metric, model và simulator không còn phụ thuộc code
object detection.

## 5. Dataset và thực nghiệm

- Synthetic: ba seed, 20 round, `N ∈ {50, 150, 200}` cho convergence chính;
  `N=100` được dùng thêm ở các figure/table scalability và ablation.
- Cross-dataset benchmark dùng SMAP (`D=25`) và MSL (`D=55`) độc lập.
- Mỗi dataset dùng cùng topology `N=100, M=10`.
- 55 SMAP source channels và 27 MSL source channels được chia thành đúng 100
  contiguous, non-overlapping normal-training shards. Không lặp sample; mỗi
  sensor shard chỉ thuộc một source channel.
- Real-data experiments dùng 30 round.

`run_experiments.py` mã hóa các suite `scalability`, `compression`, `noniid` và
`real`. Ubuntu runner chỉ gọi suite `real` cho SMAP/MSL và chạy theo thứ tự
Centralised, FedAvg, FedProx, HFL-NoCoop, HFL-Selective, HFL-Nearest.
`scripts/paper/` sinh Figure 4–8 và Table II–IV. `run_ubuntu.sh` cài môi trường,
chuẩn bị benchmark, chạy tuần tự baseline/seed và lưu raw log, round CSV,
metrics JSON, summary JSON cùng result index của phiên.

## 6. Code cũ đã loại bỏ

Các phần sau không còn thuộc bài toán mới:

- `detection_2d/` và YOLO wrappers.
- LoRA/SVD, gateway KD, teacher/student checkpoints.
- URPC/COCO sample data và data partition cho object detection.
- Demo object-detection UI.
- FedKDL-specific runner, plotting và archived OD tests.
- `main_trainer_od.py`.

Lịch sử Git vẫn giữ implementation FedKDL cũ để có thể audit; working tree mới
chỉ chứa code cần thiết cho `FL_Anomaly_Detection`.
