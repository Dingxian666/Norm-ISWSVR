import os
import random
import numpy as np
import pandas as pd

from tqdm import tqdm

from .dataio.log import Log
from .dataio.file import read_data, write_data
from .dataio.constants import Constants
from .utils.computation import compute_all_types
from .methods import internal_standard_normalization, met_normalize_SVR

class NormISWSVR():
    def __init__(self,
                data_path,
                sample_path,
                save_dir=None,
                remove_zeros=True,
                remove_zeros_minfrac_qc=0.8,
                remove_zeros_minfrac_nqc=0.5,
                svr_corr_mode='ALL',
                has_matched_IS=True,
                has_injection_order=True,
                use_before_IS=True,
                use_before_SVR=False,
                use_batch_IS=False,
                use_batch_SVR=True,
                use_after_IS=False,
                use_after_SVR=False,
                encoding='gbk',
                log_file_path=None):
        """
        data_path: str -> path of data
        sample_path: str -> path of sample info
        remove_zeros: bool -> whether to remove compounds of existing many zeros
        remove_zeros_minfrac_qc: float -> between 0 and 1, controlling the ratio of zero nums in QC samples when removing zeros
        remove_zeros_minfrac_nqc: float -> between 0 and 1, controlling the ratio of zero nums in nQC samples when removing zeros
        svr_corr_mode: str -> ['ALL', 'QC'], sample types when computing corr in SVR
        has_matched_IS: bool -> whether there is the best matched IS in data_path. Note: the column should be matchedIS (default)
        has_injection_order: bool -> whether there is the injection order in sample_path. Note: the column should be injection_order (default)
        use_before_IS: bool -> whether to use IS before batch-computing
        use_before_SVR: bool -> whether to use SVR before batch-computing
        use_batch_IS: bool -> whether to use IS in batch-computing
        use_batch_SVR: bool -> whether to use SVR in batch-computing
        use_after_IS: bool -> whether to use IS after batch-computing
        use_after_SVR: bool -> whether to use SVR after batch-computing
        encoding: str -> file format, for example, 'gbk', 'utf-8', et al.
        log_file_path: str -> path to save log
        """
        assert svr_corr_mode in ['ALL', 'QC'], 'svr_corr_mode must in [ALL, QC]'

        self.log = Log(log_file_path)

        self.remove_zeros = remove_zeros
        self.remove_zeros_minfrac_qc = remove_zeros_minfrac_qc
        self.remove_zeros_minfrac_nqc = remove_zeros_minfrac_nqc

        self.svr_corr_mode = svr_corr_mode
        self.has_matched_IS = has_matched_IS
        self.has_injection_order = has_injection_order

        self.use_batch = use_batch_IS or use_batch_SVR
        self.use_before_IS = use_before_IS
        self.use_before_SVR = use_before_SVR
        self.use_batch_IS = use_batch_IS
        self.use_batch_SVR = use_batch_SVR
        self.use_after_IS = use_after_IS
        self.use_after_SVR = use_after_SVR

        assert int(self.use_before_IS) + int(self.use_batch_IS) + int(self.use_after_IS) <= 1, \
            "At most one of 'use_before_IS, use_batch_IS, use_after_IS' can be True."

        self.sample_path = sample_path
        self.data_path = data_path

        self.encoding=encoding
        self.data_df = read_data(data_path, index_col=Constants.DATA_INDEX_COL, encoding=encoding)
        self.sample_df = read_data(sample_path, encoding=encoding)
        self.save_dir = save_dir

        self.__check_data()
        self.__pre_processing()
    
    def __check_data(self):
        # check sample_df
        null_indexes = self.sample_df[self.sample_df.isnull().values==True].index
        if len(null_indexes) > 0:
           raise ValueError(f'There are NAN values in {null_indexes} of {self.sample_path}')
        if Constants.SAMPLE_NAME_COL not in self.sample_df.columns:
            raise ValueError(f'{Constants.SAMPLE_NAME_COL} must be contained in {self.sample_path}')
        if Constants.SAMPLE_CLASS_COL not in self.sample_df.columns:
            raise ValueError(f'{Constants.SAMPLE_CLASS_COL} must be contained in {self.sample_path}')
        if self.use_batch and Constants.SAMPLE_BATCH_COL not in self.sample_df.columns:
            raise ValueError(f'{Constants.SAMPLE_BATCH_COL} must be contained in {self.sample_path}, when use_batch is {self.use_batch}')
        if self.has_injection_order and Constants.SAMPLE_INJECTION_ORDER_COL not in self.sample_df.columns:
            raise ValueError(f'{Constants.SAMPLE_INJECTION_ORDER_COL} must be contained in {self.sample_path}, when has_injection_order is {self.has_injection_order}')
        sample_names = self.sample_df[Constants.SAMPLE_NAME_COL]
        if len(sample_names) != len(set(sample_names)):
            raise ValueError(f'There are duplicate sample names in {self.sample_path}')
        classes = self.sample_df[Constants.SAMPLE_CLASS_COL].unique()
        if Constants.SAMPLE_CLASS_QC_NAMES not in classes:
            raise ValueError(f'Not exist {Constants.SAMPLE_CLASS_QC_NAMES} in {self.sample_path}.{Constants.SAMPLE_CLASS_COL}')
        # check data_df
        null_indexes = self.data_df[self.data_df.isnull().values==True].index
        if len(null_indexes) > 0:
           raise ValueError(f'There are NAN values in {null_indexes} of {self.data_path}')
        peak_names = self.data_df.index
        if len(peak_names) != len(set(peak_names)):
            raise ValueError(f'There are duplicate peak names in {self.data_path}')
        null_names = [s_name for s_name in sample_names if s_name not in set(self.data_df.columns)]
        if len(null_names) > 0:
            raise ValueError(f'Samples <{null_names}> not existed in {self.data_path}')
        if self.has_matched_IS:
            if Constants.IS_MATCH_COL_NAME not in set(self.data_df.columns):
                raise ValueError(f'{Constants.IS_MATCH_COL_NAME} must be contained in {self.data_path}, when has_matched_IS is {self.has_matched_IS}')
            self.best_match_IS = {k:self.data_df.loc[k, Constants.IS_MATCH_COL_NAME] for k in self.data_df.index}
        else:
            self.best_match_IS = None
        if self.use_before_IS or self.use_batch_IS or self.use_after_IS:
            IS_rows = [v for v in peak_names if v[:2] == Constants.IS_ROW_START_TWO_CHAR]
            if len(IS_rows) == 0:
                raise ValueError(f"No IS rows existed in {self.data_path}")
        self.log.info('Data check done!')
    
    def __pre_processing(self):
        # remove useless columns
        use_columns, useless_columns = [], []
        for v in self.data_df.columns:
            if v in list(self.sample_df[Constants.SAMPLE_NAME_COL]):
                use_columns.append(v)
            else:
                useless_columns.append(v)
        self.useless_df = self.data_df[useless_columns]
        self.data_df = self.data_df[use_columns]
        # remove peaks containing plenty of zeros
        if self.remove_zeros:
            QC_col_names = list(self.sample_df[self.sample_df[Constants.SAMPLE_CLASS_COL] == \
                Constants.SAMPLE_CLASS_QC_NAMES][Constants.SAMPLE_NAME_COL]) 
            nQC_col_names = list(self.sample_df[self.sample_df[Constants.SAMPLE_CLASS_COL] != \
                Constants.SAMPLE_CLASS_QC_NAMES][Constants.SAMPLE_NAME_COL])

            qc_df, nqc_df = self.data_df[QC_col_names], self.data_df[nQC_col_names]

            qc_idx = ((qc_df != 0).sum(axis=1) / qc_df.shape[1]) >= self.remove_zeros_minfrac_qc
            nqc_idx = ((nqc_df != 0).sum(axis=1) / nqc_df.shape[1]) >= self.remove_zeros_minfrac_nqc
            self.data_df = self.data_df[qc_idx & nqc_idx]
            compute_all_types(self.data_df, self.sample_df, logger=self.log, meta="Remove Zeros")
            if self.save_dir:
                write_data(self.data_df, os.path.join(self.save_dir, f'remove_zeros.csv'), other_df=self.useless_df)
        self.log.info('Preprocessing Done!')

    def iswsvr(self,
                data=None,
                label_df=None,
                svr_top_corr_k=5,
                svr_gamma='auto',
                svr_C=1.0,
                meta_info="ISWSVR"):
        """
        Args:
            data: pd.DataFrame -> data_df
            label_df: pd.DataFrame -> sample_df
            svr_top_corr_k: int -> the number of most related peaks used in computing corr
            svr_gamma: float or 'auto' -> the hyperparameter 'gamma' in SVR
            svr_C: float -> the hyperparameter 'C' in SVR
            meta_info: str -> for printing
        
        Returns:
            final_rsd: dict -> {'class_type': rsd_value}
        """
        assert svr_top_corr_k > 0 or self.has_injection_order, f'when has_injection_order is False, svr_top_corr_k must be > 0'
        self.log.info(f"corr_mode:{self.svr_corr_mode}\t meta: {meta_info}\n")
        
        if data is None:
            data = self.data_df
        if label_df is None:
            label_df = self.sample_df

        if self.has_injection_order:
            sample_orders = {r[Constants.SAMPLE_NAME_COL]: r[Constants.SAMPLE_INJECTION_ORDER_COL] \
                for _, r in label_df.iterrows()}

        QC_sample_names = list(label_df[label_df[Constants.SAMPLE_CLASS_COL] == \
            Constants.SAMPLE_CLASS_QC_NAMES][Constants.SAMPLE_NAME_COL]) 
        nQC_sample_names = list(label_df[label_df[Constants.SAMPLE_CLASS_COL] != \
            Constants.SAMPLE_CLASS_QC_NAMES][Constants.SAMPLE_NAME_COL])
        
        compute_all_types(data, label_df, logger=self.log, meta="Raw Data")

        if self.use_before_IS:
            data,_ = internal_standard_normalization(data, QC_col_names=QC_sample_names, best_match_IS=self.best_match_IS, IS_save=False)
            compute_all_types(data, label_df, logger=self.log, meta="IS-Before")
            if self.save_dir:
                write_data(data, os.path.join(self.save_dir, f'{meta_info}-IS-Before.csv'), other_df=self.useless_df)

        if self.use_before_SVR:
            data = met_normalize_SVR(data, injection_orders=sample_orders, QC_col_names=QC_sample_names, \
                corr_mode=self.svr_corr_mode, top_corr_k=svr_top_corr_k, gamma=svr_gamma, C=svr_C)
            compute_all_types(data, label_df, logger=self.log, meta="SVR-Before")
            if self.save_dir:
                write_data(data, os.path.join(self.save_dir, f'{meta_info}-SVR-Before.csv'), other_df=self.useless_df)

        if self.use_batch:
            # in batch
            batch_data = []
            batch_ids = label_df['batch'].unique()
            for c_bid in batch_ids:
                cur_sample_df = label_df[label_df['batch'] == c_bid]
                cur_sample_names = cur_sample_df[Constants.SAMPLE_NAME_COL]
                cur_data = data[cur_sample_names]

                cur_qc_cols = list(cur_sample_df[cur_sample_df[Constants.SAMPLE_CLASS_COL] == Constants.SAMPLE_CLASS_QC_NAMES][Constants.SAMPLE_NAME_COL])
                compute_all_types(cur_data, cur_sample_df, logger=self.log, meta=f'Raw-Batch-{c_bid}')
                if self.use_batch_IS:
                    cur_data,_ = internal_standard_normalization(cur_data, QC_col_names=cur_qc_cols, best_match_IS=self.best_match_IS,  IS_save=False)
                    compute_all_types(cur_data, cur_sample_df, logger=self.log, meta=f'IS-Batch-{c_bid}')
                if self.use_batch_SVR:
                    # svr
                    cur_data = met_normalize_SVR(cur_data, injection_orders=sample_orders, QC_col_names=cur_qc_cols,\
                        corr_mode=self.svr_corr_mode, top_corr_k=svr_top_corr_k, gamma=svr_gamma, C=svr_C)
                    compute_all_types(cur_data, cur_sample_df, logger=self.log, meta=f'SVR-Batch-{c_bid}')
                batch_data.append(cur_data)
            # 汇总结果 
            data_qc_median = np.median(data[QC_sample_names], axis=1)
            data = pd.concat(batch_data, axis=1)[data.columns]
            for i, idx in enumerate(data.index):
                data.loc[idx] = data.loc[idx] * data_qc_median[i]
            compute_all_types(data, label_df, logger=self.log, meta=f'Gather-Batch')
            if self.save_dir:
                write_data(data, os.path.join(self.save_dir, f'{meta_info}-Batch.csv'), other_df=self.useless_df)
            
        if self.use_after_IS:
            data,_ = internal_standard_normalization(data, QC_col_names=QC_sample_names, best_match_IS=self.best_match_IS, IS_save=False)
            compute_all_types(data, label_df, logger=self.log, meta=f'IS-After')
            if self.save_dir:
                write_data(data, os.path.join(self.save_dir, f'{meta_info}-IS-After.csv'), other_df=self.useless_df)
        if self.use_after_SVR:
            # svr
            data = met_normalize_SVR(data, injection_orders=sample_orders, QC_col_names=QC_sample_names, \
                corr_mode=self.svr_corr_mode, top_corr_k=svr_top_corr_k, gamma=svr_gamma, C=svr_C)
            # 汇总结果
            data_qc_median = np.median(data[QC_sample_names], axis=1)
            for i, idx in enumerate(data.index):
                data.loc[idx] = data.loc[idx] * data_qc_median[i]
            compute_all_types(data, label_df, logger=self.log, meta=f'SVR-After')
            if self.save_dir:
                write_data(data, os.path.join(self.save_dir, f'{meta_info}-SVR-After.csv'), other_df=self.useless_df)

        final_rsd = compute_all_types(data, label_df, logger=self.log, meta=f'Final')
        return final_rsd
    
    def run_with_cv_folds(self,
                        fold_nums=5,
                        svr_top_corr_k=5,
                        svr_gamma='auto',
                        svr_C=1.0,
                        meta_info=""):
        """
        Args:
            fold_nums: int -> nums of cross validation
            svr_top_corr_k: int -> the number of most related peaks used in computing corr
            svr_gamma: float or 'auto' -> the hyperparameter 'gamma' in SVR
            svr_C: float -> the hyperparameter 'C' in SVR
            meta_info: str -> for printing
        
        Returns:
            mean_results: dict -> {'class_type': the mean of rsd_value in $fold_nums folds}
        """
        assert fold_nums >= 1, "fold_nums should >= 1"
        final_results = []
        if fold_nums == 1:
            # No cross_validation
            cur_rsd = self.iswsvr(self.data_df,
                                  self.sample_df,
                                  svr_top_corr_k=svr_top_corr_k,
                                  svr_gamma=svr_gamma,
                                  svr_C=svr_C,
                                  meta_info=f'{meta_info}_ISWSVR')
            final_results.append(cur_rsd)
        else:
            raw_qc_sample_df = self.sample_df[self.sample_df[Constants.SAMPLE_CLASS_COL] == Constants.SAMPLE_CLASS_QC_NAMES].reset_index()
            raw_class_names = raw_qc_sample_df[Constants.SAMPLE_CLASS_COL]
            raw_batch_ids = raw_qc_sample_df[Constants.SAMPLE_BATCH_COL].unique()
            for cur_fold in range(fold_nums):
                # select "1 / cv_folds" data as valid dataset
                val_name_dict = {} # {old_name: new_name}
                all_valid_idx = []
                for b_id in raw_batch_ids:
                    cur_sample_df = raw_qc_sample_df[raw_qc_sample_df[Constants.SAMPLE_BATCH_COL] == b_id]
                    QC_idx = cur_sample_df[cur_sample_df[Constants.SAMPLE_CLASS_COL] == Constants.SAMPLE_CLASS_QC_NAMES].index
                    cur_valid_idx = [QC_idx[v] for v in range(cur_fold, len(QC_idx), fold_nums)]
                    all_valid_idx.extend(cur_valid_idx)
                all_valid_idx = set(all_valid_idx)
                cur_sample_df = raw_qc_sample_df
                cur_sample_df['class'] = [raw_class_names.iloc[v] if v not in all_valid_idx else 'VAL' \
                    for v in range(len(raw_class_names))]
                # 给变量换名字
                data = self.data_df[cur_sample_df[Constants.SAMPLE_NAME_COL]]
                # run
                cur_rsd = self.iswsvr(data=data,
                                      label_df=cur_sample_df,
                                      svr_top_corr_k=svr_top_corr_k,
                                      svr_gamma=svr_gamma,
                                      svr_C=svr_C,
                                      meta_info=f'{meta_info}_fold-{cur_fold}')
                final_results.append(cur_rsd)
        mean_results = {}
        for k in final_results[0].keys():
            mean_results[k] = np.mean([v[k] for v in final_results])
        return mean_results
    
    def run_with_grid_search(self,
                            fold_nums=2,
                            result_save_path='grid_search_results.csv',
                            svr_top_corr_k=[1,2,3,4,5],
                            svr_gamma=['auto', 0.1, 0.2, 0.4, 0.5],
                            svr_C=[0.1, 0.2, 0.5, 1.0],
                            tuned_times=10):
        """
        Args:
            fold_nums: int -> nums of cross validation
            result_save_path: str -> path to save the result table
            svr_top_corr_k: list -> [the number of most related peaks used in computing corr]
            svr_gamma: list -> [the hyperparameter 'gamma' in SVR]
            svr_C: list -> [the hyperparameter 'C' in SVR]
            tuned_times: int -> tuned times
        
        Returns:
            grid_df: pd.DataFrame -> the table of saving tuned results
        """
        tmp_save_dir = self.save_dir
        self.save_dir = None

        has_debuged = []
        if os.path.exists(result_save_path):
            # load old results
            grid_df = pd.read_csv(result_save_path)
            for i in range(len(grid_df)):
                has_debuged.append([grid_df.iloc[i][k] for k in ['svr_top_corr_k', 'svr_gamma', 'svr_C']])
            self.log.warning(f"Results of parameters <{has_debuged}> has existed in {result_save_path}, these parameter groups will be skipped.")
        else:
            grid_df = None

        tuned_times = min(tuned_times, len(svr_top_corr_k) * len(svr_gamma) * len(svr_C) - len(has_debuged))
        cur_tune_time = 1
        while cur_tune_time <= tuned_times:
            try:
                cur_svr_top_corr_k = random.choice(svr_top_corr_k)
                cur_svr_gamma = random.choice(svr_gamma)
                cur_svr_C = random.choice(svr_C)
                cur_debuged = [cur_svr_top_corr_k, cur_svr_gamma, cur_svr_C]
                if cur_debuged in has_debuged:
                    continue
                cur_res = self.run_with_cv_folds(fold_nums=fold_nums,
                                                svr_top_corr_k=cur_svr_top_corr_k,
                                                svr_gamma=cur_svr_gamma,
                                                svr_C=cur_svr_C,
                                                meta_info=f"tuned_idx_{cur_tune_time}")
                # handle results
                if grid_df is None:
                    grid_df = pd.DataFrame(columns = ['svr_top_corr_k', 'svr_gamma', 'svr_C'] + list(cur_res.keys()))
                grid_df.loc[len(grid_df)] = cur_debuged + list(cur_res.values())
            except Exception as e:
                self.log.error(str(e))
            finally:
                cur_tune_time += 1
        
        grid_df.to_csv(result_save_path, encoding=self.encoding, index=None)
        self.save_dir = tmp_save_dir
        return grid_df
    
if __name__ == '__main__':
    norm_iswsvr = NormISWSVR(data_path='/Users/iyuge2/Documents/github/NormISWSVR/examples/MatchedIS/demo-has-MatchedIS.csv',
                            sample_path='/Users/iyuge2/Documents/github/NormISWSVR/examples/MatchedIS/sample-info.csv',
                            save_dir='/Users/iyuge2/Documents/github/NormISWSVR/examples/results', 
                            use_after_SVR=True)
    norm_iswsvr.run_with_cv_folds()
