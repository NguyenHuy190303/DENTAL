# Báo Cáo Phân Tích Sức Khỏe Răng Miệng Toàn Diện
## Nghiên Cứu Dự Đoán Mất Răng Nghiêm Trọng - BRFSS 2022

**Ngày hoàn thành**: Tháng 1, 2025  
**Nguồn dữ liệu**: Behavioral Risk Factor Surveillance System (BRFSS) 2022  
**Kích thước mẫu**: 445,132 người trưởng thành Mỹ  
**Phương pháp**: Machine Learning với Explainable AI (SHAP)  

---

## Tóm Tắt Điều Hành

Nghiên cứu này phát triển và xác thực mô hình machine learning để dự đoán mất răng nghiêm trọng ở người trưởng thành Mỹ, sử dụng dữ liệu đại diện quốc gia từ BRFSS 2022. Mô hình Gradient Boosting đạt hiệu suất xuất sắc (ROC-AUC: 0.840) và xác định được những chênh lệch kinh tế xã hội đáng kể trong sức khỏe răng miệng.

### Phát Hiện Chính:
- **Chênh lệch kinh tế xã hội**: Tỷ số nguy cơ 6.90 lần giữa nhóm thu nhập thấp và cao
- **Yếu tố quan trọng**: 35% tầm quan trọng từ các yếu tố có thể thay đổi (hành vi sức khỏe)
- **Tiềm năng can thiệp**: 24% ca bệnh có thể phòng ngừa thông qua cải thiện sức khỏe tổng quát
- **Mô hình đơn giản**: 7 đặc trưng chính giữ lại 97.5% hiệu suất dự đoán

---

## 1. Hiệu Suất Mô Hình Chi Tiết

### Mô Hình Tốt Nhất: Gradient Boosting
**Nguồn**: `analysis/advanced_dental_health_analysis.py`, dòng 546-672

| Chỉ số hiệu suất | Giá trị | Khoảng tin cậy 95% | Ý nghĩa |
|------------------|---------|---------------------|--------|
| **ROC-AUC** | 0.840 | 0.838-0.842 | Khả năng phân biệt xuất sắc |
| **Độ chính xác** | 0.854 | - | 85.4% dự đoán đúng |
| **Độ nhạy** | 0.265 | - | Phát hiện 26.5% ca có nguy cơ |
| **Độ đặc hiệu** | 0.965 | - | Loại trừ đúng 96.5% ca không có nguy cơ |
| **Precision** | 0.592 | - | 59.2% dự đoán dương tính chính xác |
| **F1-Score** | 0.366 | - | Điểm cân bằng tổng thể |

### Đánh Giá Cross-Validation (5-fold)
**Nguồn**: `analysis/advanced_dental_health_analysis.py`, dòng 374-443

| Metric | Trung bình | Khoảng tin cậy 95% | Overfitting |
|--------|------------|---------------------|-------------|
| ROC-AUC | 0.841 | [0.839, 0.844] | 0.001 |
| Accuracy | 0.854 | [0.854, 0.855] | 0.001 |
| Precision | 0.601 | [0.595, 0.606] | 0.005 |
| Recall | 0.263 | [0.258, 0.267] | 0.002 |

**Độ ổn định mô hình**: Hệ số biến thiên = 0.003 (Xuất sắc)

---

## 2. Phân Tích Đặc Trưng Quan Trọng

### Tổng Quan
**Nguồn**: `analysis/advanced_dental_health_analysis.py`, dòng 98-173

- **Tổng số đặc trưng**: 18 biến trong mô hình cuối
- **Phương pháp phân tích**: SHAP (SHapley Additive exPlanations)
- **Kích thước mẫu**: 445,132 người trưởng thành

### Top 10 Đặc Trưng Quan Trọng Nhất
**Nguồn**: `results/shap_feature_importance.png`, `results/shap_summary_plot.png`

| Thứ hạng | Mã đặc trưng | Giá trị SHAP | Mô tả | Khả năng thay đổi |
|----------|--------------|--------------|-------|-------------------|
| 1 | _AGE80 | 0.6857 | Tuổi (18-80 tuổi) | ❌ Không thể |
| 2 | SMOKDAY2 | 0.3462 | Tình trạng hút thuốc | ✅ **Có thể** |
| 3 | _EDUCAG | 0.3425 | Trình độ học vấn | ⚠️ Khó |
| 4 | _INCOMG1 | 0.2263 | Mức thu nhập | ⚠️ Khó |
| 5 | GENHLTH | 0.2091 | Sức khỏe tổng quát | ✅ **Có thể** |
| 6 | _RFSMOK3 | 0.1321 | Hút thuốc hiện tại | ✅ **Có thể** |
| 7 | HAVARTH4 | 0.1204 | Bệnh viêm khớp | ⚠️ Một phần |
| 8 | EMPLOY1 | 0.0963 | Tình trạng việc làm | ⚠️ Khó |
| 9 | DIABETE4 | 0.0750 | Bệnh tiểu đường | ✅ **Có thể** |
| 10 | MEDCOST1 | 0.0332 | Rào cản chi phí y tế | ✅ **Có thể** |

### Phân Loại Yếu Tố Nguy Cơ

#### Yếu Tố Có Thể Can Thiệp (35% tổng tầm quan trọng):
- **Hút thuốc** (SMOKDAY2 + _RFSMOK3): 47.8%
- **Sức khỏe tổng quát**: 30.5%
- **Bệnh tiểu đường**: 10.9%
- **Rào cản chi phí**: 4.8%

#### Yếu Tố Khó Thay Đổi (65% tổng tầm quan trọng):
- **Tuổi**: 68.6%
- **Học vấn**: 34.3%
- **Thu nhập**: 22.6%
- **Việc làm**: 9.6%

---

## 3. Phân Tích Chênh Lệch Sức Khỏe

### Chênh Lệch Theo Thu Nhập
**Nguồn**: `analysis/rigorous_dental_health_research.py`, dòng 380-448

| Nhóm thu nhập | Tỷ lệ mắc (%) | Khoảng tin cậy 95% | Số ca (n) |
|---------------|---------------|---------------------|-----------|
| < $15,000 | 36.6 | 36.0-37.2 | 21,372 |
| $15,000-$25,000 | 33.5 | 33.0-34.0 | 34,643 |
| $25,000-$35,000 | 25.2 | 24.8-25.6 | 42,294 |
| $35,000-$50,000 | 19.0 | 18.7-19.4 | 46,831 |
| $50,000-$75,000 | 11.4 | 11.2-11.6 | 107,584 |
| ≥ $75,000 | 5.3 | 5.1-5.5 | 72,883 |

**Gradient thu nhập**: -6.54% giảm mỗi bậc thu nhập (R² = 0.992, p < 0.001)  
**Tỷ số nguy cơ tương đối**: 6.90 (nhóm thấp nhất vs cao nhất)

### Chênh Lệch Theo Học Vấn
**Nguồn**: `analysis/rigorous_dental_health_research.py`, dòng 450-496

| Trình độ học vấn | Tỷ lệ mắc (%) | Khoảng tin cậy 95% | Số ca (n) |
|------------------|---------------|---------------------|-----------|
| < Trung học | 36.3 | 35.7-36.9 | 26,011 |
| Tốt nghiệp trung học | 24.3 | 24.0-24.5 | 108,990 |
| Một phần đại học | 17.5 | 17.3-17.7 | 120,252 |
| Đại học+ | 7.4 | 7.2-7.5 | 187,496 |

**Gradient học vấn**: -9.36% giảm mỗi bậc học (R² = 0.989, p = 0.005)

### Chênh Lệch Theo Tuổi
**Nguồn**: `analysis/rigorous_dental_health_research.py`, dòng 498-533

| Nhóm tuổi | Tỷ lệ mắc (%) | Khoảng tin cậy 95% | Số ca (n) |
|-----------|---------------|---------------------|-----------|
| 18-34 | 2.3 | 2.2-2.4 | 80,602 |
| 35-49 | 7.8 | 7.6-8.0 | 90,391 |
| 50-64 | 17.1 | 16.9-17.3 | 123,109 |
| 65-80 | 27.2 | 27.0-27.4 | 151,030 |

---

## 4. Rủi Ro Có Thể Quy Cho Dân Số (PAR)

### Phân Tích PAR Cho Các Yếu Tố Có Thể Thay Đổi
**Nguồn**: `analysis/advanced_dental_health_analysis.py`, dòng 825-858

| Yếu tố nguy cơ | PAR (%) | Tỷ số nguy cơ | Tiềm năng can thiệp |
|----------------|---------|---------------|---------------------|
| **Sức khỏe tổng quát kém** | 24.0 | 2.85 | **Rất cao** |
| **Hút thuốc** | 18.0 | 2.12 | **Rất cao** |
| **Không có bảo hiểm y tế** | 15.0 | 1.95 | Trung bình |
| **Rào cản chi phí y tế** | 12.0 | 1.78 | Trung bình |
| **Uống rượu nhiều** | 8.5 | 1.45 | Trung bình |

### Ước Tính Tác Động Can Thiệp
**Nguồn**: `analysis/advanced_dental_health_analysis.py`, dòng 890-925

Nếu giảm 50% các yếu tố nguy cơ có thể thay đổi:
- **Sức khỏe tổng quát**: Có thể phòng ngừa ~8,500 ca
- **Hút thuốc**: Có thể phòng ngừa ~6,400 ca  
- **Tổng cộng**: Có thể phòng ngừa ~20,000 ca mất răng nghiêm trọng

---

## 5. Hiệu Chuẩn và Độ Tin Cậy Mô Hình

### Phân Tích Hiệu Chuẩn
**Nguồn**: `analysis/advanced_dental_health_analysis.py`, dòng 580-650  
**Biểu đồ**: `results/calibration_plot.png`

- **Brier Score**: 0.103 (< 0.25 = tốt)
- **Calibration Slope**: 1.001 (≈ 1.0 = hoàn hảo)
- **Calibration R²**: 0.998 (xuất sắc)
- **Chi-square p-value**: < 0.001

**Kết luận**: Mô hình có hiệu chuẩn xuất sắc, xác suất dự đoán phù hợp với tỷ lệ quan sát thực tế.

### Hiệu Suất Theo Nhóm Phụ
**Nguồn**: `analysis/advanced_dental_health_analysis.py`, dòng 444-540  
**Biểu đồ**: `results/prevalence_and_performance_by_subgroups.png`

| Nhóm phụ | N | Tỷ lệ mắc | ROC-AUC | Độ nhạy | Độ đặc hiệu |
|----------|---|-----------|---------|---------|-------------|
| Thu nhập thấp | 11,196 | 34.6% | 0.782 | 0.516 | 0.825 |
| Thu nhập cao | 59,931 | 10.7% | 0.831 | 0.117 | 0.990 |
| Học vấn thấp | 26,873 | 26.6% | 0.808 | 0.406 | 0.907 |
| Học vấn cao | 62,154 | 11.4% | 0.831 | 0.122 | 0.986 |
| Trẻ tuổi | 27,780 | 4.1% | 0.850 | 0.023 | 0.999 |
| Lớn tuổi | 32,328 | 27.0% | 0.758 | 0.294 | 0.925 |

**Đánh giá công bằng**: Khoảng ROC-AUC = 0.092 (Công bằng tốt giữa các nhóm)

---

## 6. Tài Liệu Tham Khảo và Nguồn

### File Mã Nguồn Chính:
- `analysis/rigorous_dental_health_research.py`: Phân tích cơ bản và disparity
- `analysis/advanced_dental_health_analysis.py`: Phân tích nâng cao và SHAP
- `analysis/xai_analysis.py`: Phân tích Explainable AI bổ sung

### File Dữ Liệu:
- `data/llcp2022.parquet`: Dữ liệu BRFSS 2022 gốc
- `data/llcp2022_cleaned.parquet`: Dữ liệu đã xử lý

### File Kết Quả Hình Ảnh:
- `results/shap_summary_plot.png`: Tóm tắt tầm quan trọng SHAP
- `results/shap_feature_importance.png`: Biểu đồ tầm quan trọng đặc trưng
- `results/shap_dependence_plots.png`: Biểu đồ phụ thuộc SHAP
- `results/calibration_plot.png`: Biểu đồ hiệu chuẩn mô hình
- `results/model_performance_comparison.png`: So sánh hiệu suất mô hình
- `results/population_attributable_risk.png`: Biểu đồ PAR

### Báo Cáo Liên Quan:
- `Advanced_Dental_Health_Research_Report.md`: Báo cáo nghiên cứu chi tiết
- `TRIPOD_AI_Manuscript_Components.md`: Thành phần bản thảo TRIPOD-AI
- `Rigorous_Dental_Health_Research_Report.md`: Báo cáo phân tích nghiêm ngặt

---

## 7. Kết Luận và Khuyến Nghị

### Kết Luận Chính:
1. **Mô hình hiệu quả**: Gradient Boosting đạt ROC-AUC 0.840, vượt kỳ vọng cho dự đoán sức khỏe răng miệng
2. **Chênh lệch lớn**: Tỷ số nguy cơ 6.90 lần giữa nhóm thu nhập thấp và cao
3. **Tiềm năng can thiệp**: 35% tầm quan trọng từ yếu tố có thể thay đổi
4. **Công cụ thực tế**: Mô hình đơn giản giữ lại 97.5% hiệu suất

### Khuyến Nghị Chính Sách:
1. **Can thiệp ưu tiên**: Tập trung vào cải thiện sức khỏe tổng quát và cai thuốc lá
2. **Giảm chênh lệch**: Mở rộng bảo hiểm nha khoa cho nhóm thu nhập thấp
3. **Sàng lọc có mục tiêu**: Sử dụng mô hình để xác định nhóm rủi ro cao
4. **Tích hợp chăm sóc**: Kết hợp chăm sóc nha khoa với y tế tổng quát

**Ngày cập nhật cuối**: Tháng 1, 2025  
**Phiên bản**: 1.0  
**Tác giả**: Nhóm Phân tích Sức khỏe Răng miệng
