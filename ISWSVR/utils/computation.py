import numpy as np

from ..dataio import Constants

def list_div(list1, list2):
    new_list = [float(list1[i]) / float(list2[i]) for i in range(len(list1))]
    return new_list

def compute_rsd(list_v):
    return np.std(list_v, ddof=1) / (np.mean(list_v) + 1e-8)

def compute_rsd_df(data):
    """
    功能：
        统计各个变量的rsd值
    参数：
        data: pd.DataFrame -> 要处理的数据（行为化合物，列为样本），数值类型为float
    """
    return np.std(data, ddof=1, axis=1) / (np.mean(data, axis=1) + 1e-8)

def remove_zeros_peaks(data, QC_col_names, nQC_col_names, minfrac_qc=0.8, minfrac_nqc=0.5):
    """去除零值过多的化合物
    """
    qc_df, nqc_df = data[QC_col_names], data[nQC_col_names]

    qc_idx = ((qc_df != 0).sum(axis=1) / qc_df.shape[1]) >= minfrac_qc
    nqc_idx = ((nqc_df != 0).sum(axis=1) / nqc_df.shape[1]) >= minfrac_nqc
    data = data[qc_idx & nqc_idx]
    return data

def compute_all_types(new_data, sample_df, logger, meta=""):
    # 计算QC训练结果的RSD与VALID的RSD
    sample_types = sample_df[Constants.SAMPLE_CLASS_COL].unique()
    final_rsd = {}
    for cur_type in sample_types:
        cur_data = new_data[sample_df[sample_df[Constants.SAMPLE_CLASS_COL] == cur_type]['sample_name']]
        final_rsd[cur_type] = np.median(compute_rsd_df(cur_data))
    logger.info(f"{meta}: {final_rsd}")
    return final_rsd