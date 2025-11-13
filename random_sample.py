import numpy as np
import pandas as pd
import sys
from log_code import setup_logging
logger = setup_logging('random_sample')

def random_value_filling(x_train, x_test):
    try:
        i = []  # Column to fill
        for i in x_train.columns:
            if x_train[i].isnull().sum() != 0:
                logger.info(f"Column Name: {i} | Missing Values Before: {x_train[i].isnull().sum()}")

                # Random sample imputation for train data
                r = x_train[i].dropna().sample(x_train[i].isnull().sum(), random_state=42)
                r.index = x_train[x_train[i].isnull()].index
                x_train.loc[x_train[i].isnull(), i] = r

                logger.info(f"col names : {i} and null values count : {x_train[i].isnull().sum()}")

                logger.info(f"Column Name: {i} | Missing Values Before: {x_test[i].isnull().sum()}")

                # Random sample imputation for test data
                r_test = x_train[i].dropna().sample(x_test[i].isnull().sum(), random_state=42)
                r_test.index = x_test[x_test[i].isnull()].index
                x_test.loc[x_test[i].isnull(), i] = r_test

                logger.info(f"col names : {i} and null values count : {x_test[i].isnull().sum()}")



    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to {er_msg}')
