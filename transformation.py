'''
here we are implementing log transformation concept
'''

import numpy as np
import pandas as pd
import sklearn
import sys
import os
import logging
from log_code import setup_logging
logger = setup_logging('transformation')
import warnings
warnings.filterwarnings("ignore")

def log_tran(X_train_num,X_test_num):
    try:
        for i in X_train_num.columns:
            X_train_num[i+'_log'] = np.log(X_train_num[i]+1)
            X_test_num[i + '_log'] = np.log(X_test_num[i] + 1)
        logger.info(f'Log Transformation completed successfully : {X_train_num.columns}')

        f = []
        for j in X_train_num.columns:
            if '_log' not in j:
                f.append(j)
        logger.info(f'Log column names : {f}')
        X_train_num = X_train_num.drop(f, axis=1)
        X_test_num = X_test_num.drop(f, axis=1)
        return X_train_num,X_test_num

    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to {er_msg}')


