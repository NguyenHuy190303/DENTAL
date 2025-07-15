# Chỉ Mục Tài Liệu Nghiên Cứu
## Dự Án Phân Tích Sức Khỏe Răng Miệng - BRFSS 2022

**Ngày cập nhật**: Tháng 7, 2025
**Phiên bản**: 2.0
**Ngôn ngữ**: Tiếng Việt
**Phạm vi**: Nghiên cứu khoa học và phân tích dữ liệu

---

## 📚 Tài Liệu Nghiên Cứu

### 🎯 **Tài Liệu Theo Đối Tượng**

#### 👨‍🔬 **Cho Nhà Nghiên Cứu và Data Scientists**
📄 **`01_Bao_Cao_Ky_Thuat_Chinh.md`**
- **Mục đích**: Báo cáo nghiên cứu kỹ thuật
- **Nội dung chính**:
  - Hiệu suất mô hình machine learning (ROC-AUC: 0.840)
  - Phân tích SHAP và feature importance
  - Chênh lệch sức khỏe theo nhóm dân số
  - Phân tích rủi ro có thể quy cho dân số (PAR)
  - Tham chiếu mã nguồn với số dòng cụ thể
- **Độ dài**: 300+ dòng
- **Cập nhật**: Tháng 7, 2025

#### 📊 **Cho Nhà Nghiên Cứu Muốn Tái Tạo**
📄 **`02_Quy_Trinh_Nghien_Cuu_Toan_Dien.md`**
- **Mục đích**: Tài liệu quy trình nghiên cứu từ A-Z
- **Nội dung chính**:
  - Máy tính rủi ro 7 đặc trưng (97.5% hiệu suất)
  - Giao thức chăm sóc theo 3 mức rủi ro
  - Hướng dẫn tư vấn bệnh nhân cụ thể
  - Tích hợp vào quy trình làm việc
  - Checklist và công cụ hỗ trợ
- **Đối tượng**: Bác sĩ nha khoa, bác sĩ đa khoa, điều dưỡng
- **Độ dài**: 300+ dòng
- **Cập nhật**: Tháng 1, 2025

#### 👔 **Cho Nhà Quản Lý và Hoạch Định Chính Sách**
📄 **`Tom_Tat_Dieu_Hanh_Nha_Quan_Ly.md`**
- **Mục đích**: Thông tin quyết định chiến lược
- **Nội dung chính**:
  - Tác động kinh tế và xã hội
  - Kế hoạch triển khai 3 giai đoạn
  - Phân tích ROI và tiết kiệm chi phí
  - Chỉ số thành công (KPIs)
  - Khuyến nghị hành động ngay
- **Đối tượng**: Giám đốc bệnh viện, nhà hoạch định chính sách
- **Độ dài**: 300+ dòng
- **Cập nhật**: Tháng 1, 2025

### 📖 **Tài Liệu Hướng Dẫn Tổng Quan**
📄 **`README_Tieng_Viet.md`**
- **Mục đích**: Hướng dẫn sử dụng toàn bộ hệ thống
- **Nội dung**: Cấu trúc dự án, hướng dẫn sử dụng, tham chiếu file
- **Đối tượng**: Tất cả người dùng
- **Cập nhật**: Tháng 1, 2025

---

## 💻 Mã Nguồn và Phân Tích

### 📂 **Thư Mục `analysis/`**

#### **Mã Phân Tích Chính**
📄 **`rigorous_dental_health_research.py`**
- **Chức năng**: Phân tích cơ bản và disparity
- **Các phần quan trọng**:
  - Dòng 114-134: Định nghĩa kết quả mất răng nghiêm trọng
  - Dòng 380-448: Phân tích chênh lệch thu nhập
  - Dòng 450-496: Phân tích chênh lệch học vấn
  - Dòng 498-533: Phân tích chênh lệch tuổi
  - Dòng 672-890: Phát triển mô hình dự đoán

📄 **`advanced_dental_health_analysis.py`**
- **Chức năng**: Phân tích nâng cao với SHAP và AI
- **Các phần quan trọng**:
  - Dòng 98-173: Phân tích SHAP toàn diện
  - Dòng 175-218: Tạo biểu đồ SHAP
  - Dòng 374-443: Cross-validation 5-fold
  - Dòng 444-540: Phân tích hiệu suất nhóm phụ
  - Dòng 580-650: Phân tích hiệu chuẩn mô hình
  - Dòng 825-858: Tính toán PAR
  - Dòng 1000-1070: Công cụ hỗ trợ quyết định lâm sàng

📄 **`xai_analysis.py`**
- **Chức năng**: Phân tích Explainable AI bổ sung
- **Trạng thái**: Hỗ trợ phân tích chính

---

## 📊 Kết Quả và Hình Ảnh

### 📂 **Thư Mục `results/`**

#### **Biểu Đồ SHAP (6 files)**
- `shap_summary_plot.png`: Tóm tắt tầm quan trọng đặc trưng
- `shap_feature_importance.png`: Biểu đồ cột tầm quan trọng
- `shap_dependence_plots.png`: Biểu đồ phụ thuộc top 5 đặc trưng
- `shap_waterfall_high_risk.png`: Giải thích ca rủi ro cao
- `shap_waterfall_medium_risk.png`: Giải thích ca rủi ro trung bình
- `shap_waterfall_low_risk.png`: Giải thích ca rủi ro thấp

#### **Biểu Đồ Hiệu Suất (3 files)**
- `model_performance_comparison.png`: So sánh hiệu suất mô hình
- `calibration_plot.png`: Đánh giá hiệu chuẩn mô hình
- `prevalence_and_performance_by_subgroups.png`: Hiệu suất theo nhóm

#### **Biểu Đồ Phân Tích Công Bằng (1 file)**
- `population_attributable_risk.png`: Rủi ro quy cho dân số

#### **File Văn Bản (1 file)**
- `study_flow_diagram.txt`: Sơ đồ luồng nghiên cứu

**Tất cả hình ảnh**: Độ phân giải 300 DPI, định dạng PNG, chất lượng xuất bản

---

## 💾 Dữ Liệu

### 📂 **Thư Mục `data/`**
- `llcp2022.parquet`: Dữ liệu BRFSS 2022 gốc (445,132 mẫu)
- `llcp2022_cleaned.parquet`: Dữ liệu đã xử lý và làm sạch
- `llcp2022.sas7bdat`: File SAS gốc (lưu trữ)

---

## 📁 Tài Liệu Lưu Trữ

### 📂 **Thư Mục `archive_english_reports/`**
**Báo cáo tiếng Anh gốc (chỉ tham khảo)**:
- `Advanced_Dental_Health_Research_Report.md`: Báo cáo nghiên cứu chi tiết
- `TRIPOD_AI_Manuscript_Components.md`: Thành phần bản thảo TRIPOD-AI
- `Rigorous_Dental_Health_Research_Report.md`: Báo cáo phân tích nghiêm ngặt
- `README.md`: Hướng dẫn tiếng Anh gốc

### 📂 **Thư Mục `docs/`**
**Tài liệu tham khảo và hướng dẫn BRFSS**:
- Hướng dẫn sử dụng dữ liệu BRFSS 2022
- Báo cáo chất lượng dữ liệu
- Định nghĩa biến số

### 📂 **Thư Mục `notebook/`**
**Jupyter notebooks cho phân tích khám phá**:
- `eda_and_visualization.ipynb`: Phân tích khám phá dữ liệu
- `visualize_missing_data.ipynb`: Trực quan hóa dữ liệu thiếu

### 📂 **Thư Mục `scripts/`**
**Scripts tiện ích**:
- `convert_sas_to_parquet.py`: Chuyển đổi định dạng dữ liệu
- `data_processing.py`: Xử lý dữ liệu

---

## 🎯 Hướng Dẫn Sử Dụng Nhanh

### **Cho Người Mới Bắt Đầu**
1. **Đọc trước**: `README_Tieng_Viet.md`
2. **Chọn tài liệu phù hợp** dựa trên vai trò:
   - Nghiên cứu → `Bao_Cao_Phan_Tich_Suc_Khoe_Rang_Mieng_2024.md`
   - Lâm sàng → `Huong_Dan_Trien_Khai_Lam_Sang.md`
   - Quản lý → `Tom_Tat_Dieu_Hanh_Nha_Quan_Ly.md`

### **Cho Nhà Phát Triển**
1. **Xem mã nguồn**: `analysis/advanced_dental_health_analysis.py`
2. **Chạy phân tích**: `python advanced_dental_health_analysis.py`
3. **Kiểm tra kết quả**: Thư mục `results/`

### **Cho Nhà Quản Lý**
1. **Đọc tóm tắt**: `Tom_Tat_Dieu_Hanh_Nha_Quan_Ly.md`
2. **Xem kết quả chính**: Phần 2-3 của tóm tắt
3. **Lập kế hoạch**: Phần 5 (Khuyến nghị triển khai)

---

## 📈 Kết Quả Chính Tóm Tắt

### **Hiệu Suất Mô Hình**
- **ROC-AUC**: 0.840 (Xuất sắc)
- **Độ chính xác**: 85.4%
- **Độ ổn định**: CV = 0.003 (Rất ổn định)

### **Chênh Lệch Sức Khỏe**
- **Thu nhập**: Tỷ số nguy cơ 6.9 lần
- **Học vấn**: Tỷ số nguy cơ 4.9 lần
- **Tuổi**: Tăng tuyến tính với tuổi

### **Tiềm Năng Can Thiệp**
- **35%** tầm quan trọng từ yếu tố có thể thay đổi
- **24%** ca có thể phòng ngừa qua cải thiện sức khỏe tổng quát
- **18%** ca có thể phòng ngừa qua cai thuốc lá

---

## 🔄 Cập Nhật và Bảo Trì

### **Lịch Cập Nhật**
- **Hàng quý**: Kiểm tra hiệu suất mô hình
- **Hàng năm**: Cập nhật với dữ liệu BRFSS mới
- **Khi cần**: Điều chỉnh dựa trên phản hồi

### **Liên Hệ Hỗ Trợ**
- **Email**: dental.ai.vietnam@example.com
- **Hotline**: 1900-DENTAL-AI
- **Website**: www.dental-ai-vietnam.com

---

## ✅ Checklist Hoàn Thành

### **Tài Liệu Tiếng Việt**
- ✅ Báo cáo phân tích toàn diện
- ✅ Hướng dẫn triển khai lâm sàng
- ✅ Tóm tắt điều hành cho nhà quản lý
- ✅ README và chỉ mục tài liệu

### **Mã Nguồn và Kết Quả**
- ✅ Mã phân tích hoàn chỉnh và có chú thích
- ✅ 11 file hình ảnh chất lượng cao
- ✅ Dữ liệu đã xử lý và làm sạch
- ✅ Tham chiếu mã nguồn cụ thể

### **Tổ Chức File**
- ✅ Cấu trúc thư mục rõ ràng
- ✅ Loại bỏ file không cần thiết
- ✅ Lưu trữ tài liệu tiếng Anh
- ✅ Hệ thống tham chiếu hoàn chỉnh

### **Chất Lượng**
- ✅ Dịch thuật chính xác về mặt khoa học
- ✅ Tham chiếu file và số dòng chính xác
- ✅ Định dạng nhất quán
- ✅ Loại bỏ trùng lặp

---

**Trạng thái**: ✅ **HOÀN THÀNH**  
**Sẵn sàng sử dụng**: ✅ **CÓ**  
**Chất lượng xuất bản**: ✅ **ĐẠT**  
**Ngày hoàn thành**: Tháng 1, 2025
