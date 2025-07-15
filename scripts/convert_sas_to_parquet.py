import pandas as pd
from pathlib import Path

def convert_sas_to_parquet(sas_file_path, parquet_file_path):
    """
    Chuyển đổi tệp dữ liệu từ định dạng SAS (.sas7bdat) sang định dạng Parquet (.parquet).

    Args:
        sas_file_path (str): Đường dẫn đến tệp SAS đầu vào.
        parquet_file_path (str): Đường dẫn đến tệp Parquet đầu ra.
    """
    print(f"🔄 Bắt đầu quá trình chuyển đổi...")
    
    # Kiểm tra xem tệp SAS có tồn tại không
    if not Path(sas_file_path).exists():
        print(f"❌ Lỗi: Không tìm thấy tệp đầu vào '{sas_file_path}'")
        return

    try:
        # Đọc tệp SAS vào một pandas DataFrame
        print(f"📄 Đang đọc tệp SAS: '{sas_file_path}'...")
        df = pd.read_sas(sas_file_path, format='sas7bdat', encoding='ISO-8859-1')
        print(f"✅ Đọc tệp SAS thành công. Dữ liệu có {df.shape[0]} dòng và {df.shape[1]} cột.")

        # Ghi DataFrame ra tệp Parquet
        print(f"✍️ Đang ghi ra tệp Parquet: '{parquet_file_path}'...")
        df.to_parquet(parquet_file_path, engine='pyarrow')
        print(f"✅ Chuyển đổi thành công! Dữ liệu đã được lưu tại '{parquet_file_path}'")

    except Exception as e:
        print(f"❌ Đã xảy ra lỗi trong quá trình chuyển đổi: {e}")

if __name__ == '__main__':
    # Định nghĩa đường dẫn tệp
    input_sas_file = 'llcp2022.sas7bdat'
    output_parquet_file = 'llcp2022_converted.parquet'

    # Thực hiện chuyển đổi
    convert_sas_to_parquet(input_sas_file, output_parquet_file) 