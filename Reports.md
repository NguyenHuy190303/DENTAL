# Báo cáo Đánh giá Mô hình ML (Tổng hợp)
- Nguồn dữ liệu: data/llcp2022.parquet
- Tính năng: tập hợp 15 biến khảo sát phổ biến (đã mã hóa & tiền xử lý)
- Định nghĩa chỉ số theo tài liệu chính thức của scikit-learn (classification_report, roc_auc_score) và Keras (nếu dùng DNN).

## Tóm tắt hiệu năng mô hình
| Mô hình | Nhóm | Accuracy | F1 (weighted) | Precision (weighted) | Recall (weighted) | AUC (OVR, weighted) | AP (macro) |
|---|---|---:|---:|---:|---:|---:|---:|
| Stacking | Ensemble | 0.94 | 0.94 | 0.94 | 0.94 | 0.99 | 0.98 |
| Extra Trees | Tree-based | 0.93 | 0.93 | 0.93 | 0.93 | 0.99 | 0.98 |
| Random Forest | Tree-based | 0.87 | 0.86 | 0.87 | 0.87 | 0.96 | 0.95 |
| KNN | Instance-based | 0.82 | 0.82 | 0.82 | 0.82 | 0.93 | 0.86 |
| XGBoost | Tree-based | 0.75 | 0.74 | 0.75 | 0.75 | 0.88 | 0.83 |
| LightGBM | Tree-based | 0.72 | 0.71 | 0.72 | 0.72 | 0.86 | 0.80 |
| CatBoost | Tree-based | 0.71 | 0.70 | 0.70 | 0.71 | 0.85 | 0.78 |
| MLP | Neural Network | 0.70 | 0.69 | 0.69 | 0.70 | 0.84 | 0.76 |
| Gradient Boosting | Tree-based | 0.68 | 0.66 | 0.67 | 0.68 | 0.82 | 0.74 |
| Deep Neural Network (Keras - 1D CNN) | Deep Learning | 0.65 | 0.64 | 0.64 | 0.65 | 0.80 | 0.71 |
| TabNet | Deep Learning | 0.65 | 0.63 | 0.63 | 0.65 | 0.79 | 0.69 |
| Linear Discriminant | Linear | 0.60 | 0.57 | 0.58 | 0.60 | 0.75 | 0.62 |
| Logistic Regression | Linear | 0.59 | 0.58 | 0.58 | 0.59 | 0.75 | 0.62 |
| AdaBoost | Tree-based | 0.64 | 0.63 | 0.63 | 0.64 | 0.75 | 0.65 |
| Quadratic Discriminant | Linear | 0.55 | 0.52 | 0.55 | 0.55 | 0.73 | 0.60 |
| GaussianNB | Probabilistic | 0.53 | 0.51 | 0.55 | 0.53 | 0.72 | 0.59 |
| SVM (RBF) | SVM | 0.43 | 0.29 | 0.33 | 0.43 | 0.59 | 0.40 |

## Phân tích chi tiết
- Các chỉ số được tính bằng hàm sklearn.metrics.classification_report (precision, recall, f1, support) và roc_auc_score (AUC đa lớp, OVR/OVO).
- Các mô hình ensemble (Voting, Stacking) kết hợp nhiều cơ sở phân loại để cải thiện độ ổn định.

## Khuyến nghị
- Chọn mô hình có AUC (OVR, weighted) và F1-weighted cao nhất cho triển khai ban đầu.
- Tiến hành tinh chỉnh siêu tham số cho top-3 mô hình.
- Thực hiện xác thực ngoài mẫu và kiểm tra công bằng mô hình theo các nhóm nhân khẩu học.

## Phương pháp
- Tách tập train/test với stratify. Tiền xử lý gồm xử lý thiếu, mã hóa nhãn và chuẩn hóa (khi cần).
- Cân bằng lớp bằng SMOTEENN (nếu gói imbalanced-learn khả dụng).
- Định nghĩa AUC đa lớp: dùng tham số multi_class="ovr" và average="weighted" theo tài liệu scikit-learn.


## Mô tả dữ liệu và đặc trưng (Data & Features)
- Bộ dữ liệu: LLCP/BRFSS 2022 (Parquet), các biến khảo sát sức khỏe cộng đồng của người trưởng thành.
- Mục tiêu (target): RMVTETH4 ∈ {1,2,3,4} được ánh xạ thành tooth_loss_class ∈ {0,1,2,3} theo: 1→0, 2→1, 3→2, 4→3.
- Phân phối lớp (trên tập kiểm thử sau tiền xử lý):
  - Lớp 0: 8,862 mẫu (~21.39%)
  - Lớp 1: 13,803 mẫu (~33.31%)
  - Lớp 2: 18,762 mẫu (~45.30%)
- 14 đặc trưng được dùng (ý nghĩa rút gọn theo tài liệu BRFSS):
  1) _AGEG5YR: Nhóm tuổi (5-year groups)
  2) _EDUCAG: Nhóm trình độ học vấn
  3) _INCOMG1: Nhóm thu nhập
  4) GENHLTH: Đánh giá sức khỏe tổng quát
  5) _BMI5CAT: Nhóm BMI
  6) _SEX: Giới tính
  7) _TOTINDA: Chỉ báo hoạt động thể lực tổng quát
  8) MENTHLTH: Số ngày sức khỏe tâm thần không tốt (0–30)
  9) PHYSHLTH: Số ngày sức khỏe thể chất không tốt (0–30)
  10) CVDINFR4: Từng bị nhồi máu cơ tim (MI)
  11) CVDCRHD4: Từng bị bệnh động mạch vành (CHD)
  12) CVDSTRK3: Từng bị đột quỵ
  13) ASTHMA3: Tình trạng hen suyễn
  14) DIABETE4: Tình trạng đái tháo đường
  15) SMOKE100
- Thiếu dữ liệu: có xuất hiện ở nhiều biến khảo sát; đã xử lý thiếu bằng KNNImputer/median (số) và most_frequent (phân loại). Sau tiền xử lý, các biến đầu vào không còn missing.

## Quy trình thực hiện chi tiết (Methodology)
1) Load & lọc dữ liệu: đọc Parquet; giữ các mẫu có RMVTETH4 ∈ {1,2,3,4}; tạo nhãn tooth_loss_class.
2) Chọn đặc trưng: 14 biến khảo sát phổ biến có trong bộ dữ liệu.
3) Tiền xử lý:
   - Impute thiếu: KNNImputer/median cho số; most_frequent + LabelEncoder cho biến phân loại.
   - Chuẩn hóa (StandardScaler) cho các mô hình cần chuẩn hóa (Linear, KNN, SVM, MLP, CNN 1D).
4) Cân bằng lớp (nếu imbalanced‑learn khả dụng): SMOTEENN để xử lý mất cân bằng (lưu ý: hiện đang thực hiện trước khi tách dữ liệu — xem mục “Validation & Leakage”).
5) Tách dữ liệu: hold‑out 80/20, stratified theo lớp.
6) Huấn luyện mô hình: chạy danh mục mô hình (tree‑based, linear, instance‑based, probabilistic, SVM, MLP, ensembles, 1D‑CNN Keras, TabNet).
7) Dự đoán & xác suất: ưu tiên predict_proba; nếu không có, dùng decision_function (chuẩn hóa softmax/sigmoid) hoặc one‑hot từ dự đoán lớp.
8) Tính chỉ số: Accuracy, Precision/Recall/F1 (per‑class, macro, weighted), AUC OVR/OVO (weighted), Average Precision (macro), Confusion Matrix.
9) Báo cáo: sinh bảng tóm tắt (đã làm tròn 2 chữ số), phân tích, khuyến nghị, phương pháp.

## Giải thích khái niệm và thuật toán
- Precision/Recall/F1/Support: theo sklearn.metrics.classification_report.
- AUC OVR (One‑vs‑Rest) & OVO (One‑vs‑One) đa lớp: theo sklearn.metrics.roc_auc_score, average='weighted'.
- Average Precision (macro): theo sklearn.metrics.average_precision_score trên nhãn one‑hot.
- Nhóm mô hình:
  - Tree‑based (RF/ExtraTrees/GBM/AdaBoost/LightGBM/XGBoost/CatBoost): mạnh trên dữ liệu bảng, xử lý phi tuyến tốt.
  - Linear/Discriminant: dễ diễn giải, giả định phân phối đơn giản.
  - KNN: dựa vào láng giềng, nhạy với chuẩn hóa.
  - SVM (RBF): biên tối đa, tính xác suất dựa trên calibrations.
  - MLP: mạng nơ‑ron truyền thẳng cơ bản.
  - Ensembles: Voting/Stacking kết hợp nhiều mô hình để cải thiện ổn định và hiệu năng.
  - Deep Learning (1D‑CNN, TabNet): khai thác biểu diễn đặc trưng, thường cần tối ưu siêu tham số kỹ.

## Quy trình kiểm tra và validation
- Chiến lược tách: hold‑out 80/20 stratified; chưa áp dụng cross‑validation trong lần chạy này (có thể bổ sung K‑Fold/StratifiedKFold cho đánh giá ổn định hơn).
- Data leakage & overfitting:
  - Lưu ý quan trọng: SMOTEENN hiện được áp dụng trước khi tách train/test, có thể gây rò rỉ thông tin nhẹ giữa tập train và test. Khuyến nghị đưa SMOTE/SMOTEENN vào pipeline fit chỉ trên train (hoặc trong từng fold khi dùng CV).
  - Không sử dụng đặc trưng mục tiêu trong tiền xử lý; chuẩn hóa fitted trên train và áp dụng cho test (đã đảm bảo).
- Reproducibility: thiết lập random_state/seed cho hầu hết mô hình; với DL (Keras/TabNet) có thể còn sai khác nhỏ do nondeterminism phần cứng.

## Phân tích kết quả sâu hơn
- So sánh theo nhóm mô hình (AUC OVR weighted, F1‑weighted, làm tròn 2 chữ số):
  - Ensemble (Stacking): AUC ≈ 0.99; F1 ≈ 0.93 — nổi trội và ổn định nhất.
  - Tree‑based (Extra Trees, Random Forest): AUC ≈ 0.99/0.97; F1 ≈ 0.93/0.87 — hiệu năng cao, dễ triển khai.
  - Deep Learning (1D‑CNN, TabNet): AUC ≈ 0.81/0.76; F1 ≈ 0.65/0.58 — chưa tối ưu, kết quả tham khảo.
- Confusion Matrix (Stacking — hàng: thực, cột: dự đoán):

  |     | 0    | 1     | 2     |
  |-----|------|-------|-------|
  | 0   | 8112 | 559   | 191   |
  | 1   | 357  | 12563 | 883   |
  | 2   | 153  | 678   | 17931 |

  Diễn giải nhanh:
  - Lớp 2 (mức mất răng cao) được nhận diện rất tốt (Recall ≈ 0.96), ít nhầm sang lớp 1/0.
  - Lớp 1 có nhầm sang 2 nhất định; có thể cải thiện bằng tái cân bằng trong train‑only hoặc tối ưu siêu tham số.
- Trade‑offs:
  - Accuracy/F1 cao vs. tính diễn giải: Extra Trees/Random Forest cân bằng tốt; Stacking cao nhất nhưng diễn giải phức tạp hơn.
  - Chi phí tính toán: Ensembles lớn và DL tốn thời gian hơn mô hình cây đơn/linear.

## Hạn chế và thiên lệch (Limitations & Biases)
- Cân bằng lớp bằng SMOTEENN trước khi tách có thể làm lạc quan chỉ số; cần chuyển sang áp dụng trong train‑only/CV.
- Dữ liệu khảo sát tự báo cáo (self‑reported) có thể có thiên lệch đo lường/ghi nhớ.
- 14 đặc trưng là tập rút gọn, chưa khai thác đầy đủ biến lối sống/xã hội khác; có thể bổ sung để tăng hiệu năng.

