import os
import pandas as pd

def read_data(data_path, index_col=None, encoding='gbk'):
    _, file_extension = os.path.splitext(data_path)
    if file_extension not in ['.csv', '.excel']:
        raise TypeError(f"Not support {file_extension} file format.")
    if file_extension == '.csv':
        df = pd.read_csv(data_path, index_col=index_col, encoding=encoding)
    else:
        df = pd.read_excel(data_path, index_col=index_col, encoding=encoding)
    return df

def write_data(data, save_path, encoding='gbk', other_df=None):
    if other_df is not None:
        other_df = other_df.loc[data.index]
    data = pd.concat([other_df, data], axis=1)
    data.to_csv(save_path, encoding=encoding)
    return