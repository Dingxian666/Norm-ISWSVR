import os
import pandas as pd
import numpy as np

from tqdm import tqdm

from ..utils.computation import list_div, compute_rsd, compute_rsd_df
from ..dataio.constants import Constants

def internal_standard_normalization(data, QC_col_names=None, best_match_IS=None, IS_save=False):
    """
    功能:
        对每个非内标化合物，从预设的内标化合物中选取最佳内标对其进行矫正
    参数:
        data: pd.DataFrame -> 要处理的数据（行为化合物，行名为化合物名字，内标化合物都在最前面，且名字以IS开头；列为样本，列名为样本名字，QC样本命名以QC开头）；数值类型为float
        best_match_IS: None or dict -> {peak_name: best_match_IS_name}
        IS_save: bool -> save IS rows ?
    返回值：
        res_df: pd.DataFrame -> 矫正后的数据
        ret_best_IS: dict -> {化合物名：矫正使用的IS行}
    """
    peak_names = list(data.index)
    # confirm cols of IS
    if best_match_IS is not None:
        # check IS
        IS_rows = set(best_match_IS.values())
        index = set(data.index)
        for v in IS_rows:
            if v not in index:
                raise ValueError(f"Illegal IS names in '{best_match_IS}'!")
    else:
        IS_rows = [v for v in peak_names if v[:2] == Constants.IS_ROW_START_TWO_CHAR]
        if len(IS_rows) == 0:
            raise ValueError("No IS rows existed...")
        if len(set(IS_rows)) != len(IS_rows):
            raise ValueError("Exist repeated IS rows...")
        IS_rows = set(IS_rows)

    # {'IS2': v(QCs)}
    IS_MEAN_dict, IS_dict = {}, {} 
    for v in IS_rows:
        IS_dict[v] = list(data.loc[v, QC_col_names])
        IS_MEAN_dict[v] = np.mean(IS_dict[v])
    # {'IS2': v(alls)}
    ALL_IS_MEAN_dict, ALL_IS_dict = {}, {} 
    for v in IS_rows:
        ALL_IS_dict[v] = list(data.loc[v])
        ALL_IS_MEAN_dict[v] = np.mean(ALL_IS_dict[v])
    
    # start normalization
    results = []
    ret_best_IS = {}
    no_IS_rows = [v for v in peak_names if v not in IS_rows]
    for cur_peak in tqdm(no_IS_rows):
        cur_QCs = list(data.loc[cur_peak, QC_col_names])
        cur_alls = list(data.loc[cur_peak])
        # RSD value before normalization
        before_RSD = compute_rsd(cur_QCs)
        before_all_RSD = compute_rsd(cur_alls)
        if best_match_IS is not None:
            best_IS = best_match_IS[cur_peak]
            after_RSD = -1
        else:
            # match
            min_QCs_RSD = 1000000
            best_IS = ''
            for k, v in IS_dict.items():
                # var / IS * mean(IS)
                QCs_var_is = list_div(cur_QCs, v)
                QCs_var_is = [v * IS_MEAN_dict[k] for v in QCs_var_is]

                ALL_var_is = list_div(cur_alls, ALL_IS_dict[k])
                ALL_var_is = [v * ALL_IS_MEAN_dict[k] for v in ALL_var_is]

                new_all_RSD = compute_rsd(ALL_var_is)
                QCs_RSD = compute_rsd(QCs_var_is)
                if new_all_RSD < before_all_RSD and QCs_RSD < min_QCs_RSD:
                    best_IS = k
                    min_QCs_RSD = QCs_RSD
            after_RSD = min_QCs_RSD
        # RSD value after normalization
        if before_RSD <= after_RSD:
            # don't need normalization
            results.append(list(data.loc[cur_peak]))
            ret_best_IS[cur_peak] = '#'
        else:
            # normalization ({M / IS * (IS_mean)})
            temp_v = list_div(list(data.loc[cur_peak]), list(data.loc[best_IS]))
            temp_v = [v * ALL_IS_MEAN_dict[best_IS] for v in temp_v]
            # save
            results.append(temp_v)
            ret_best_IS[cur_peak] = best_IS
    res_df = pd.DataFrame(results, index=no_IS_rows, columns=data.columns)
    if IS_save:
        res_df = pd.concat([data.loc[IS_rows], res_df], axis=0)
    return res_df, ret_best_IS
