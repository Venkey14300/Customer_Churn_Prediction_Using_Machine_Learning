import numpy as np
import pandas as pd
import sklearn
import sys
import os
import logging
from log_code import setup_logging
logger = setup_logging('constant')
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_selection import VarianceThreshold
reg = VarianceThreshold(threshold=0)

def constant_technique(train_num,test_num):
    try:
        reg.fit(train_num)
        logger.info(f'Total columns : {train_num.shape[1]} -> without variance 0 : {sum(reg.get_support())} -> with variance 0 : {sum(~reg.get_support())}')
        # logger.info(f'Variance 0 : names : {train_num.columns[reg.get_support()]}')
        logger.info(f'Variance 0 : names : {train_num.columns[~reg.get_support()]}')
        # if 'SeniorCitizen_log_trim' in train_num.columns:
        #     logger.info(f"SeniorCitizen_log_trim value counts: {train_num['SeniorCitizen_log_trim'].value_counts()}")

        # zeros = (test_num['SeniorCitizen_log_trim'] == 0).sum()
        # ones = (test_num['SeniorCitizen_log_trim'] == 1).sum()
        # logger.info(f'0s: {zeros}, 1s: {ones}')

        train_num = train_num.drop(['SeniorCitizen_log_trim'], axis=1)
        test_num = test_num.drop(['SeniorCitizen_log_trim'], axis=1)

        logger.info(f'Train column names : {train_num.columns}')
        logger.info(f'Test column names : {test_num.columns}')
        return train_num,test_num


    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to {er_msg}')