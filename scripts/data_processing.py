import pandas as pd
import numpy as np

# Đường dẫn đến file dữ liệu của bạn
file_path = 'llcp2022.sas7bdat'

print(f"Bắt đầu đọc file: {file_path}...")
print("Lưu ý: Quá trình này có thể mất vài phút do kích thước file lớn (1.1 GB).")

try:
    # Đọc file .sas7bdat bằng pandas
    # Thêm encoding='ISO-8859-1' để tránh lỗi liên quan đến ký tự
    df = pd.read_sas(file_path, format='sas7bdat', encoding='ISO-8859-1')
    
    print("\n✅ Tải dữ liệu thành công!")
    
    # --- PHÂN TÍCH KHÁM PHÁ BAN ĐẦU (INITIAL EDA) ---
    
    # 1. Hiển thị thông tin cơ bản của DataFrame
    print("\n--- Thông tin tổng quan về dữ liệu ---")
    print(f"Số dòng (bệnh nhân): {df.shape[0]}")
    print(f"Số cột (thuộc tính): {df.shape[1]}")
    
    # 2. Xem 5 dòng đầu tiên để hình dung dữ liệu
    print("\n--- 5 dòng dữ liệu đầu tiên ---")
    print(df.head())
    
    # 3. Kiểm tra tỷ lệ dữ liệu bị thiếu (missing data)
    print("\n--- Phân tích dữ liệu bị thiếu (Missing Data Analysis) ---")
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    missing_info = pd.DataFrame({
        ' cột': df.columns,
        'tỷ lệ thiếu (%)': missing_percentage
    })
    
    # Sắp xếp theo tỷ lệ thiếu giảm dần
    missing_info = missing_info.sort_values(by='tỷ lệ thiếu (%)', ascending=False)
    
    print("Tỷ lệ phần trăm dữ liệu bị thiếu ở mỗi cột:")
    # In ra các cột có dữ liệu bị thiếu
    print(missing_info[missing_info['tỷ lệ thiếu (%)'] > 0].to_string())

    # 4. Xác định các cột cần loại bỏ (theo yêu cầu > 15% missing)
    columns_to_drop_df = missing_info[missing_info['tỷ lệ thiếu (%)'] > 15]
    list_of_cols_to_drop = columns_to_drop_df[' cột'].tolist()

    print(f"\n--- Các cột đề xuất loại bỏ (thiếu > 15%) ---")
    if list_of_cols_to_drop:
        print(f"Tìm thấy {len(list_of_cols_to_drop)} cột để loại bỏ.")
        # for col_name in list_of_cols_to_drop:
        #     print(f"- {col_name} ({missing_percentage[col_name]:.2f}%)")
    else:
        print("Không có cột nào có tỷ lệ thiếu > 15%.")

    # --- BƯỚC 2: THỰC HIỆN LÀM SẠCH VÀ LƯU KẾT QUẢ ---
    
    print(f"\nBắt đầu loại bỏ {len(list_of_cols_to_drop)} cột...")
    df_cleaned = df.drop(columns=list_of_cols_to_drop)
    print(f"✅ Đã loại bỏ. Kích thước dữ liệu mới: {df_cleaned.shape}")

    # Lưu dữ liệu đã làm sạch vào định dạng Parquet để truy cập nhanh hơn
    output_path = 'llcp2022_cleaned.parquet'
    print(f"\nLưu dữ liệu đã xử lý vào file: {output_path}")
    try:
        df_cleaned.to_parquet(output_path)
        print(f"✅ Đã lưu thành công vào {output_path}")
    except ImportError:
        print("❌ Lỗi: Cần cài đặt thư viện 'pyarrow'. Hãy chạy: pip install pyarrow")
    
    # Phân tích dữ liệu thiếu trên các cột còn lại
    print("\n--- Phân tích dữ liệu thiếu trên các cột còn lại (sau khi đã xóa) ---")
    if not df_cleaned.empty:
        remaining_missing = (df_cleaned.isnull().sum() / len(df_cleaned)) * 100
        remaining_missing_info = remaining_missing[remaining_missing > 0].sort_values(ascending=False)
        
        if remaining_missing_info.empty:
            print("Chúc mừng! Không còn dữ liệu bị thiếu trong các cột còn lại.")
        else:
            print("Tỷ lệ thiếu của các cột còn lại:")
            print(remaining_missing_info.to_string())

except FileNotFoundError:
    print(f"❌ LỖI: Không tìm thấy file tại đường dẫn: {file_path}")
except Exception as e:
    print(f"❌ Đã xảy ra lỗi không xác định: {e}") 