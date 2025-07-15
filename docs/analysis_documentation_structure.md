# Phân Tích Cấu Trúc Documentation Hiện Tại

## Phân Loại File Markdown

### 📋 **Báo Cáo Tiếng Việt (Giữ lại và tối ưu hóa)**

#### 1. **Báo Cáo Kỹ Thuật Chính**
- `Bao_Cao_Phan_Tich_Suc_Khoe_Rang_Mieng.md` ✅
  - Nội dung: Báo cáo kỹ thuật toàn diện với kết quả ML
  - Trạng thái: Chất lượng cao, giữ nguyên
  - Đề xuất tên mới: `01_Bao_Cao_Ky_Thuat_Chinh.md`

#### 2. **Báo Cáo Quy Trình Nghiên Cứu**
- `BAO_CAO_TOAN_DIEN_QUY_TRINH_NGHIEN_CUU.md` ✅
  - Nội dung: Quy trình nghiên cứu từ A-Z
  - Trạng thái: Rất chi tiết, có thể rút gọn
  - Đề xuất tên mới: `02_Quy_Trinh_Nghien_Cuu_Toan_Dien.md`

#### 3. **Hướng Dẫn Lâm Sàng**
- `Huong_Dan_Trien_Khai_Lam_Sang.md` ✅
  - Nội dung: Hướng dẫn cho nhà cung cấp y tế
  - Trạng thái: Thực tế và hữu ích
  - Đề xuất tên mới: `03_Huong_Dan_Lam_Sang.md`

#### 4. **Tóm Tắt Điều Hành**
- `Tom_Tat_Dieu_Hanh_Nha_Quan_Ly.md` ✅
  - Nội dung: Tóm tắt cho nhà quản lý
  - Trạng thái: Cần kiểm tra và tối ưu hóa
  - Đề xuất tên mới: `04_Tom_Tat_Dieu_Hanh.md`

#### 5. **README Chính**
- `README_Tieng_Viet.md` ✅
  - Nội dung: Tổng quan dự án
  - Trạng thái: Sẽ trở thành README.md chính
  - Đề xuất: Thay thế README.md hiện tại

#### 6. **Chỉ Mục Tài Liệu**
- `Chi_Muc_Tai_Lieu_Chinh.md` ✅
  - Nội dung: Danh mục tài liệu
  - Trạng thái: Cần cập nhật theo cấu trúc mới
  - Đề xuất tên mới: `00_Chi_Muc_Tai_Lieu.md`

#### 7. **Báo Cáo Hoàn Thành**
- `Bao_Cao_Hoan_Thanh_He_Thong_Tai_Lieu.md` ✅
  - Nội dung: Báo cáo hoàn thành hệ thống
  - Trạng thái: Có thể hợp nhất với báo cáo quy trình
  - Đề xuất: Hợp nhất vào `02_Quy_Trinh_Nghien_Cuu_Toan_Dien.md`

### 🗑️ **Báo Cáo Tiếng Anh (Cần xử lý)**

#### 1. **Báo Cáo Phân Tích BRFSS**
- `docs/BRFSS_2022_Dental_Health_Analysis_Report.md` ❌
  - Nội dung: Phân tích thống kê chi tiết
  - Vấn đề: Trùng lặp với báo cáo tiếng Việt
  - Hành động: **XÓA** (đã có phiên bản tiếng Việt tốt hơn)

#### 2. **Hướng Dẫn Can Thiệp**
- `docs/Dental_Health_Intervention_Guidelines.md` ❌
  - Nội dung: Hướng dẫn can thiệp y tế công cộng
  - Vấn đề: Chưa có phiên bản tiếng Việt
  - Hành động: **DỊCH** sang tiếng Việt trước khi xóa

#### 3. **Phương Pháp Thống Kê**
- `docs/Statistical_Methods_and_Data_Quality.md` ❌
  - Nội dung: Phương pháp thống kê và chất lượng dữ liệu
  - Vấn đề: Nội dung kỹ thuật, chưa có phiên bản tiếng Việt
  - Hành động: **DỊCH** và hợp nhất vào báo cáo kỹ thuật chính

#### 4. **README Docs**
- `docs/README.md` ❌
  - Nội dung: Tổng quan thư mục docs
  - Hành động: **XÓA** và tạo mới bằng tiếng Việt

## Kế Hoạch Tổ Chức Lại

### Cấu Trúc Mới Đề Xuất:
```
docs/
├── reports/
│   ├── 00_Chi_Muc_Tai_Lieu.md
│   ├── 01_Bao_Cao_Ky_Thuat_Chinh.md
│   ├── 02_Quy_Trinh_Nghien_Cuu_Toan_Dien.md
│   ├── 03_Huong_Dan_Lam_Sang.md
│   ├── 04_Tom_Tat_Dieu_Hanh.md
│   └── 05_Huong_Dan_Can_Thiep_Y_Te.md (dịch từ tiếng Anh)
├── references/
│   ├── [PDF files đã di chuyển]
│   └── Phuong_Phap_Thong_Ke_Va_Chat_Luong_Du_Lieu.md (dịch từ tiếng Anh)
└── README.md (từ README_Tieng_Viet.md)
```

### Nguyên Tắc Đặt Tên:
- Sử dụng số thứ tự (00, 01, 02...) để sắp xếp logic
- Tên file bằng tiếng Việt, không dấu, dùng underscore
- Tên ngắn gọn nhưng mô tả rõ nội dung
- Nhất quán về format và cấu trúc
