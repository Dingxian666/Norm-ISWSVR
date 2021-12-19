import pandas as pd
import numpy as np

from sklearn import svm
from tqdm import tqdm

from .is_norm import internal_standard_normalization

def met_normalize_SVR(data,
                        QC_col_names,
                        injection_orders=None,
                        corr_mode='ALL',
                        top_corr_k=5,
                        gamma='auto',
                        C=1.0):
    """
    功能：
        参考MetNormalizer的SVR处理
    参数:
        data: pd.DataFrame -> 要处理的数据（行为化合物，行名为化合物名字，内标化合物都在最前面，且名字以IS开头；列为
        样本，列名为样本名字，QC样本命名以QC开头）；数值类型为float
    """
    assert corr_mode in ['ALL','QC']
    assert injection_orders is not None or top_corr_k > 0

    if top_corr_k > 0:
        if corr_mode == 'QC':
            corr = abs(data[QC_col_names].transpose().corr())
        elif corr_mode == 'ALL':
            corr = abs(data.transpose().corr())
    
    data_qc = data[QC_col_names]
    if injection_orders is not None:
        qc_orders = [injection_orders[name] for name in data_qc.columns]
        all_orders = [injection_orders[name] for name in data.columns]

    res_df = {}
    for peak_name in tqdm(data_qc.index):
        if top_corr_k > 0:
            cur_corr = corr.loc[peak_name].drop([peak_name], axis=0).sort_values(ascending=False)
            top_corr = cur_corr[:top_corr_k].index
            x = data_qc.loc[top_corr].transpose()
            if injection_orders is not None:
                x['order'] = qc_orders
        else:
            x = np.array(qc_orders).reshape(-1, 1)
        x_mean, x_std = x.mean(axis=0), x.std(axis=0)
        xx = (x - x_mean) / (x_std + 1e-8)

        y = data_qc.loc[peak_name]
        y_mean, y_std = y.mean(), y.std()
        yy = (y - y_mean) / (y_std + 1e-8)
        # 训练
        svr = svm.SVR(gamma=gamma, C=C)
        svr.fit(xx, yy)

        # 测试
        if top_corr_k > 0:
            x = data.loc[top_corr].transpose()
            if injection_orders is not None:
                x['order'] = all_orders
        else:
            x = np.array(all_orders).reshape(-1, 1)
        y = data.loc[peak_name]
        xx = (x - x_mean) / (x_std + 1e-8)
        y_pred = svr.predict(xx)
        y_pred = y_pred * y_std + y_mean
        res_df[peak_name] = y / y_pred
    res_df = pd.DataFrame(res_df).transpose()
    res_df.columns = data.columns
    return res_df