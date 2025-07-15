"""
Data processing utilities for dental health research.
Includes functions for data conversion, cleaning, and preprocessing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_sas_to_parquet(sas_file_path: Union[str, Path], 
                          parquet_file_path: Union[str, Path]) -> bool:
    """
    Chuyển đổi tệp dữ liệu từ định dạng SAS (.sas7bdat) sang định dạng Parquet (.parquet).

    Args:
        sas_file_path: Đường dẫn đến tệp SAS đầu vào
        parquet_file_path: Đường dẫn đến tệp Parquet đầu ra

    Returns:
        bool: True nếu chuyển đổi thành công, False nếu có lỗi
    """
    logger.info("🔄 Bắt đầu quá trình chuyển đổi...")
    
    # Kiểm tra xem tệp SAS có tồn tại không
    if not Path(sas_file_path).exists():
        logger.error(f"❌ Lỗi: Không tìm thấy tệp đầu vào '{sas_file_path}'")
        return False

    try:
        # Đọc tệp SAS vào một pandas DataFrame
        logger.info(f"📄 Đang đọc tệp SAS: '{sas_file_path}'...")
        df = pd.read_sas(sas_file_path, format='sas7bdat', encoding='ISO-8859-1')
        logger.info(f"✅ Đọc tệp SAS thành công. Dữ liệu có {df.shape[0]} dòng và {df.shape[1]} cột.")

        # Ghi DataFrame ra tệp Parquet
        logger.info(f"✍️ Đang ghi ra tệp Parquet: '{parquet_file_path}'...")
        df.to_parquet(parquet_file_path, engine='pyarrow')
        logger.info(f"✅ Chuyển đổi thành công! Dữ liệu đã được lưu tại '{parquet_file_path}'")
        return True

    except Exception as e:
        logger.error(f"❌ Đã xảy ra lỗi trong quá trình chuyển đổi: {e}")
        return False


def load_brfss_data(file_path: Union[str, Path], 
                   file_format: str = 'parquet') -> Optional[pd.DataFrame]:
    """
    Tải dữ liệu BRFSS từ file.

    Args:
        file_path: Đường dẫn đến file dữ liệu
        file_format: Định dạng file ('parquet', 'sas', 'csv')

    Returns:
        pd.DataFrame hoặc None nếu có lỗi
    """
    try:
        if file_format.lower() == 'parquet':
            df = pd.read_parquet(file_path)
        elif file_format.lower() == 'sas':
            df = pd.read_sas(file_path, format='sas7bdat', encoding='ISO-8859-1')
        elif file_format.lower() == 'csv':
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Định dạng file không được hỗ trợ: {file_format}")
        
        logger.info(f"✅ Tải dữ liệu thành công: {df.shape[0]} dòng, {df.shape[1]} cột")
        return df
    
    except Exception as e:
        logger.error(f"❌ Lỗi khi tải dữ liệu: {e}")
        return None


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Lấy thông tin tổng quan về dataset.

    Args:
        df: DataFrame cần phân tích

    Returns:
        dict: Thông tin về dataset
    """
    info = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2),
        'dtypes': df.dtypes.value_counts(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist()
    }
    return info


def clean_brfss_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch dữ liệu BRFSS cơ bản.

    Args:
        df: DataFrame gốc

    Returns:
        pd.DataFrame: DataFrame đã được làm sạch
    """
    df_clean = df.copy()
    
    # Log thông tin ban đầu
    logger.info(f"Dữ liệu ban đầu: {df_clean.shape[0]} dòng, {df_clean.shape[1]} cột")
    
    # Xử lý missing values cơ bản
    initial_missing = df_clean.isnull().sum().sum()
    logger.info(f"Tổng số missing values: {initial_missing}")
    
    # Có thể thêm các bước làm sạch cụ thể ở đây
    
    logger.info(f"Dữ liệu sau khi làm sạch: {df_clean.shape[0]} dòng, {df_clean.shape[1]} cột")
    return df_clean


if __name__ == '__main__':
    # Ví dụ sử dụng
    input_sas_file = 'data/llcp2022.sas7bdat'
    output_parquet_file = 'data/llcp2022_converted.parquet'
    
    # Thực hiện chuyển đổi
    success = convert_sas_to_parquet(input_sas_file, output_parquet_file)
    
    if success:
        # Tải và kiểm tra dữ liệu
        df = load_brfss_data(output_parquet_file)
        if df is not None:
            info = get_data_info(df)
            print(f"Thông tin dataset: {info}")
