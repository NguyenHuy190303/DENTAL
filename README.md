# DENTAL - Nghiên Cứu Phân Tích Sức Khỏe Răng Miệng
## Dự Đoán Mất Răng Nghiêm Trọng Sử Dụng Machine Learning và Explainable AI

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![BRFSS](https://img.shields.io/badge/Data-BRFSS%202022-orange.svg)](https://www.cdc.gov/brfss/)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.840-brightgreen.svg)](docs/reports/01_Bao_Cao_Ky_Thuat_Chinh.md)

**Phiên bản**: 1.0
**Ngày cập nhật**: Tháng 7, 2025
**Nguồn dữ liệu**: Behavioral Risk Factor Surveillance System (BRFSS) 2022
**Kích thước mẫu**: 445,132 người trưởng thành Mỹ
**Phương pháp**: Machine Learning với Explainable AI (SHAP)

---

## 🎯 Tổng Quan Dự Án

### Mục Tiêu Nghiên Cứu
Dự án này phát triển một hệ thống dự đoán mất răng nghiêm trọng sử dụng machine learning tiên tiến, nhằm:

🔬 **Nghiên cứu khoa học**: Phân tích dữ liệu đại diện quốc gia để hiểu các yếu tố nguy cơ
📊 **Mô hình dự đoán**: Xây dựng công cụ ML với hiệu suất cao (ROC-AUC: 0.840)
⚖️ **Phân tích công bằng**: Đánh giá chênh lệch sức khỏe theo nhóm kinh tế xã hội
🔍 **Explainable AI**: Sử dụng SHAP để giải thích quyết định của mô hình

### Phát Hiện Chính
🏆 **Hiệu suất mô hình xuất sắc**: ROC-AUC 0.840, độ chính xác 85.4%
📈 **Chênh lệch nghiêm trọng**: Tỷ số nguy cơ 6.9 lần giữa nhóm thu nhập thấp và cao
🎯 **Yếu tố can thiệp**: 35% tầm quan trọng từ các yếu tố có thể thay đổi
💡 **Tiềm năng phòng ngừa**: 24% ca bệnh có thể phòng ngừa qua cải thiện sức khỏe
🛠️ **Công cụ thực tế**: Mô hình đơn giản 7 đặc trưng giữ lại 97.5% hiệu suất

### Ý Nghĩa Khoa Học
- **Nghiên cứu lớn nhất**: Phân tích 445,132 mẫu đại diện toàn quốc Mỹ
- **Phương pháp tiên tiến**: Kết hợp ML và XAI theo chuẩn TRIPOD-AI
- **Tác động xã hội**: Xác định bất bình đẳng sức khỏe và đề xuất can thiệp
- **Reproducible research**: Code và dữ liệu được tổ chức theo chuẩn khoa học

---

## 📁 Cấu Trúc Dự Án

```
DENTAL/
├── README.md                     # Tài liệu hướng dẫn chính
├── requirements.txt              # Dependencies Python với version pinning
├── environment.yml               # Môi trường Conda
├── .gitignore                    # Git ignore rules
│
├── src/                          # 📦 Mã nguồn chính
│   ├── __init__.py
│   ├── analysis/                 # 🔬 Phân tích và mô hình ML
│   │   ├── __init__.py
│   │   ├── base.py              # Phân tích cơ bản và chênh lệch sức khỏe
│   │   ├── advanced.py          # Phân tích nâng cao với SHAP/XAI
│   │   └── xai.py               # Explainable AI bổ sung
│   ├── data/                    # 📊 Xử lý dữ liệu
│   │   ├── __init__.py
│   │   └── processing.py        # Chuyển đổi SAS, làm sạch dữ liệu
│   ├── utils/                   # 🛠️ Tiện ích chung
│   │   ├── __init__.py
│   │   └── common_imports.py    # Import chung và cấu hình
│   └── config/                  # ⚙️ Cấu hình dự án
│       ├── __init__.py
│       └── settings.py          # Thiết lập paths, parameters
│
├── tests/                       # 🧪 Unit tests
│   ├── __init__.py
│   ├── test_analysis/           # Tests cho analysis module
│   ├── test_data/               # Tests cho data module
│   └── test_utils/              # Tests cho utils module
│
├── docs/                        # 📚 Tài liệu nghiên cứu
│   ├── README.md                # Hướng dẫn sử dụng docs
│   ├── reports/                 # 📄 Báo cáo nghiên cứu (tiếng Việt)
│   │   ├── 00_Chi_Muc_Tai_Lieu.md
│   │   ├── 01_Bao_Cao_Ky_Thuat_Chinh.md
│   │   ├── 02_Quy_Trinh_Nghien_Cuu_Toan_Dien.md
│   │   └── 03_Tom_Tat_Dieu_Hanh.md
│   └── references/              # 📖 Tài liệu tham khảo
│       └── *.pdf                # PDF files từ CDC về BRFSS 2022
│
├── data/                        # 💾 Dữ liệu nghiên cứu
│   ├── llcp2022.sas7bdat        # Dữ liệu gốc BRFSS 2022 (1.1GB)
│   ├── llcp2022.parquet         # Dữ liệu đã chuyển đổi
│   └── llcp2022_cleaned.parquet # Dữ liệu đã làm sạch
│
├── results/                     # 📈 Kết quả phân tích
│   ├── *.png                    # Biểu đồ SHAP và hiệu suất mô hình
│   └── study_flow_diagram.txt   # Sơ đồ luồng nghiên cứu
│
├── notebooks/                   # 📓 Jupyter notebooks
│   ├── eda_and_visualization.ipynb
│   └── visualize_missing_data.ipynb
│
└── analysis/                    # 📁 Code cũ (legacy)
    └── ...                      # Sẽ được dọn dẹp sau
```

### � Mô Tả Thư Mục Chính

| Thư mục | Mục đích | Nội dung chính |
|---------|----------|----------------|
| **`src/`** | Mã nguồn chính của dự án | Modules Python được tổ chức theo chức năng |
| **`docs/`** | Tài liệu nghiên cứu | Báo cáo tiếng Việt và tài liệu tham khảo |
| **`data/`** | Dữ liệu BRFSS 2022 | Raw data, processed data, cleaned data |
| **`results/`** | Kết quả phân tích | Biểu đồ, metrics, outputs từ mô hình |
| **`tests/`** | Unit tests | Test cases cho các modules |
| **`notebooks/`** | Jupyter notebooks | EDA, visualization, prototyping |

---

## 🚀 Hướng Dẫn Bắt Đầu Nhanh

### 1. Cài Đặt Môi Trường

#### Sử dụng pip:
```bash
git clone https://github.com/NguyenHuy190303/DENTAL.git
cd DENTAL
pip install -r requirements.txt
```

#### Sử dụng conda:
```bash
git clone https://github.com/NguyenHuy190303/DENTAL.git
cd DENTAL
conda env create -f environment.yml
conda activate DENTAL
```

### 2. Khám Phá Dự Án

#### Đọc tài liệu:
- **Bắt đầu**: `docs/reports/00_Chi_Muc_Tai_Lieu.md`
- **Kỹ thuật**: `docs/reports/01_Bao_Cao_Ky_Thuat_Chinh.md`
- **Quy trình**: `docs/reports/02_Quy_Trinh_Nghien_Cuu_Toan_Dien.md`

#### Chạy notebooks:
```bash
jupyter notebook notebooks/eda_and_visualization.ipynb
```

### 3. Sử Dụng Code

#### Tiền xử lý dữ liệu:
```python
from src.data.processing import convert_sas_to_parquet, load_brfss_data

# Chuyển đổi SAS sang Parquet
convert_sas_to_parquet('data/llcp2022.sas7bdat', 'data/llcp2022.parquet')

# Tải dữ liệu
df = load_brfss_data('data/llcp2022.parquet')
```

#### Chạy phân tích:
```python
from src.analysis.base import RigorousDentalHealthResearch

# Phân tích cơ bản
analysis = RigorousDentalHealthResearch()
results = analysis.run_analysis(df)
```

---

## 🧪 Chạy Tests

```bash
# Chạy tất cả tests
pytest tests/

# Chạy tests cho module cụ thể
pytest tests/test_analysis/

# Chạy với coverage report
pytest --cov=src tests/
```

---

## 📊 Kết Quả Nghiên Cứu

### Hiệu Suất Mô Hình
- **ROC-AUC**: 0.840 (Xuất sắc)
- **Độ chính xác**: 85.4%
- **Chênh lệch sức khỏe**: Tỷ số nguy cơ 6.9 lần giữa nhóm thu nhập

### Outputs Chính
- **Biểu đồ SHAP**: Feature importance và explainability
- **Model performance**: So sánh các thuật toán ML
- **Health equity analysis**: Phân tích chênh lệch theo nhóm dân số
- **Population attributable risk**: Đánh giá tác động dân số

---

## 🤝 Đóng Góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

---

## 📄 License

Dự án này được phân phối dưới MIT License. Xem `LICENSE` file để biết thêm chi tiết.

---

## 📞 Liên Hệ

- **Dự án**: Nghiên cứu phân tích sức khỏe răng miệng BRFSS 2022
- **Phương pháp**: Machine Learning với Explainable AI
- **Mục tiêu**: Nghiên cứu khoa học và phân tích dữ liệu sức khỏe công cộng
