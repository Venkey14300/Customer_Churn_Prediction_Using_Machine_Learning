"""
In this Main file we are going to load data and call respected functions for model development.
"""
import pickle

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import logging

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from log_code import setup_logging
logger = setup_logging('main')
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from random_sample import random_value_filling
from transformation import log_tran
from trimming import trim_tech
from constant import constant_technique
from quasi_constant import quasi_constant_technique
from hypothesis_testing import fs
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from training_algorithms import common
from sklearn.ensemble import GradientBoostingClassifier
import pickle

class CHURN_PROJECT_INFO:
    try:
        def __init__(self, path):
            self.df = pd.read_csv(path)

            # Convert TotalCharges to numeric and handle blanks
            self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')

            logger.info(f"Data Loaded Successfully : {self.df.shape}")
            self.X = self.df.iloc[:, :-1]  # independent data
            self.y = self.df.iloc[:, -1]  # dependent data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )

            # Checking missing values before filling
            logger.info(f"Missing Values before imputation: {self.X_train['TotalCharges'].isnull().sum()}")
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to {er_msg}')

    def missing_values(self):
        try:
            # Fill missing values in TotalCharges using random sample imputation
            random_value_filling(self.X_train, self.X_test)
            logger.info(f"Missing Values after imputation: {self.X_train['TotalCharges'].isnull().sum()}")
            logger.info(f'{self.X_train.isnull().sum()} : data type : {self.X_train.dtypes}')

            # Drop customerID column from both train and test sets
            if 'customerID' in self.X_train.columns:
                self.X_train = self.X_train.drop(columns=['customerID'])
            if 'customerID' in self.X_test.columns:
                self.X_test = self.X_test.drop(columns=['customerID'])

            # separating X_train data into category and numeric
            self.X_train_num = self.X_train.select_dtypes(exclude = 'object')
            self.X_train_cat = self.X_train.select_dtypes(include='object')

            # separating X_test data into category and numeric
            self.X_test_num = self.X_test.select_dtypes(exclude='object')
            self.X_test_cat = self.X_test.select_dtypes(include='object')

            logger.info(f"Numerical Column name :{self.X_train_num.columns}")
            logger.info(f"Categorical Column name :{self.X_train_cat.columns}")


        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to {er_msg}')


    def handle_outliers(self):
        try:
            # lets convert Each numerical column into log transformation
            # 70% of outliers we can remove
            # then we can pass to trimming formula to remove all outliers in the data
            self.X_train_num,self.X_test_num = log_tran(self.X_train_num, self.X_test_num)
            self.X_train_num,self.X_test_num = trim_tech(self.X_train_num,self.X_test_num)

            logger.info(f'main column names : {self.X_train_num.columns}')
            logger.info(f'main column names : {self.X_test_num.columns}')

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to {er_msg}')



    def feature_selection(self):
        try:
            self.X_train_num,self.X_test_num = constant_technique(self.X_train_num,self.X_test_num)
            logger.info(f'Train column names : {self.X_train_num.columns}')
            logger.info(f'Test column names : {self.X_test_num.columns}')

            self.X_train_num,self.X_test_num = quasi_constant_technique(self.X_train_num,self.X_test_num)
            logger.info(f'Train column names : {self.X_train_num.columns}')
            logger.info(f'Test column names : {self.X_test_num.columns}')

            self.X_train_num,self.X_test_num = fs(self.X_train_num,self.X_test_num,self.y_train,self.y_test)
            logger.info(f'Train column names : {self.X_train_num.columns}')
            logger.info(f'Test column names : {self.X_test_num.columns}')


        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to {er_msg}')


    def cat_to_num(self):
        try:
            logger.info(f"Categorical Column name train :{self.X_train_cat.columns}")
            logger.info(f"Categorical Column name test:{self.X_test_cat.columns}")

            # applying Onehot Encoding for all the categorical Columns except Contract
            oh = OneHotEncoder(categories='auto',drop='first',handle_unknown='ignore')
            oh.fit(self.X_train_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'PaymentMethod']])
            logger.info(f'{oh.categories_}')
            logger.info(f'{oh.get_feature_names_out()}')
            res_oh = oh.transform(self.X_train_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'PaymentMethod']]).toarray()
            res_test = oh.transform(self.X_test_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'PaymentMethod']]).toarray()
            f = pd.DataFrame(res_oh, columns = oh.get_feature_names_out())
            f_test = pd.DataFrame(res_test, columns=oh.get_feature_names_out())
            self.X_train_cat.reset_index(drop = True, inplace = True)
            f.reset_index(drop = True, inplace = True)
            self.X_test_cat.reset_index(drop=True, inplace=True)
            f_test.reset_index(drop=True, inplace=True)
            self.X_train_cat = pd.concat([self.X_train_cat,f], axis = 1)
            self.X_test_cat = pd.concat([self.X_test_cat, f_test], axis=1)
            # logger.info(f'{self.X_train_cat.columns}')
            # logger.info(f'{self.X_train_cat.sample(10)}')
            # logger.info(f'{self.X_train_cat.isnull().sum()}')
            # logger.info(f'{self.X_train_cat.columns}')

            # now applying Ordinal encoding on Contract column
            od = OrdinalEncoder()
            od.fit(self.X_train_cat[['Contract']])
            logger.info(f'{od.categories_}')
            logger.info(f'columns_names : {od.get_feature_names_out()}')
            res_od = od.transform(self.X_train_cat[['Contract']])
            res_od_test = od.transform(self.X_test_cat[['Contract']])
            col_names = od.get_feature_names_out()
            f1 = pd.DataFrame(res_od, columns=col_names+['_con'])
            f1_test = pd.DataFrame(res_od_test, columns=col_names + ['_con'])
            self.X_train_cat.reset_index(drop=True, inplace=True)
            f1.reset_index(drop=True, inplace=True)
            self.X_test_cat.reset_index(drop=True, inplace=True)
            f1_test.reset_index(drop=True, inplace=True)
            self.X_train_cat = pd.concat([self.X_train_cat, f1], axis=1)
            self.X_test_cat = pd.concat([self.X_test_cat, f1_test], axis=1)
            # logger.info(f'{self.X_train_cat.columns}')
            # logger.info(f'{self.X_train_cat.sample(10)}')
            # logger.info(f'{self.X_train_cat.isnull().sum()}')
            # logger.info(f'{self.X_train_cat.columns}')

            self.X_train_cat = self.X_train_cat.drop(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'PaymentMethod', 'Contract'], axis = 1)
            self.X_test_cat = self.X_test_cat.drop(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'PaymentMethod', 'Contract'], axis=1)

            feature_cols = list(self.X_train_num.columns) + list(self.X_train_cat.columns)
            with open('C:\\Users\\LENOVO\\Documents\\PROJECT_ALL_FILES\\Result\\feature_columns.pkl', 'wb') as f:
                pickle.dump(feature_cols, f)
            logger.info(f'Feature columns saved to feature_columns.pkl : {len(feature_cols)} columns')

            logger.info(f'train data : {self.X_train_cat.isnull().sum()}')
            logger.info(f'test data : {self.X_test_cat.isnull().sum()}')

            logger.info(f'{self.X_train_cat.sample(10)}')
            logger.info(f'{self.X_test_cat.sample(10)}')

            # logger.info(f'-----------------------------------------------------------')


            # logger.info(f'{self.y_train.unique()}')
            logger.info(f'y_train_data : {self.y_train.unique()}')
            logger.info(f'y_train_data : {self.y_train.isnull().sum()}')
            logger.info(f'y_test_data : {self.y_test.unique()}')
            logger.info(f'y_test_data : {self.y_test.isnull().sum()}')
            #dependent variable should be converted using label encoder

            logger.info(f'{self.y_train[:10]}')

            lb = LabelEncoder()
            lb.fit(self.y_train)
            self.y_train = lb.transform(self.y_train)
            self.y_test = lb.transform(self.y_test)

            # logger.info(f'-----------------------------------------------------------')
            logger.info(f'detailed{lb.classes_}')
            logger.info(f'{self.y_train[:10]}')
            logger.info(f'y_train_data : {self.y_train.shape}')
            logger.info(f'y_test_data : {self.y_test.shape}')

            # here : 0 ----> No
            #        1 ----> Yes

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to {er_msg}')

    def combining_both_cat_num(self):
        try:
            # first we have to reset index then we have to concat
            self.X_train_num.reset_index(drop = True, inplace = True)
            self.X_train_cat.reset_index(drop=True, inplace=True)

            self.X_test_num.reset_index(drop=True, inplace=True)
            self.X_test_cat.reset_index(drop=True, inplace=True)

            self.training_data = pd.concat([self.X_train_num, self.X_train_cat], axis = 1)
            self.testing_data = pd.concat([self.X_test_num, self.X_test_cat], axis=1)

            logger.info(f'training data shape : {self.training_data.shape} : training data columns : {self.training_data.columns}')
            logger.info(f'testing data shape : {self.testing_data.shape} : testing data columns : {self.testing_data.columns}')

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to {er_msg}')

    def balanced_data(self):
        try:
            logger.info(f'--------------Before Data Balancing-----------------')
            logger.info(f'total rows for good category in training data {self.training_data.shape[0]} was : {sum(self.y_train == 1)}')
            logger.info(f'total rows for bad category in training data {self.training_data.shape[0]} was : {sum(self.y_train == 0)}')
            logger.info(f'--------------After Data Balancing-----------------')
            sm = SMOTE(random_state = 42)
            self.training_data_res,self.y_train_res = sm.fit_resample(self.training_data, self.y_train)
            logger.info(f'total rows for good category in training data {self.training_data_res.shape[0]} was : {sum(self.y_train_res == 1)}')
            logger.info(f'total rows for bad category in training data {self.training_data_res.shape[0]} was : {sum(self.y_train_res == 0)}')

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to {er_msg}')


    def feature_scaling(self):
        try:
            logger.info(f'--------------Before Feature Scaling-----------------')
            logger.info(f'{self.training_data_res.head(5)}')

            sc = StandardScaler()
            sc.fit(self.training_data_res)
            self.training_data_res_transformation = sc.transform(self.training_data_res)
            self.testing_data_transformation = sc.transform(self.testing_data)

            with open('C:\\Users\\LENOVO\\Documents\\PROJECT_ALL_FILES\\Result\\standard_scaler.pkl', 'wb') as t:
                pickle.dump(sc, t)

            logger.info('-----------After Feature Scaling-----------------')
            logger.info(f'{self.training_data_res_transformation}')

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to {er_msg}')


    # def train_models(self):
    #     try:
    #         common(self.training_data_res_transformation,self.y_train_res,self.testing_data_transformation,self.y_test)
    #
    #     except Exception as e:
    #         er_ty, er_msg, er_lin = sys.exc_info()
    #         logger.info(f'Issue is : {er_lin.tb_lineno} : due to {er_msg}')

    def best_model(self):
        try:
            logger.info(f'--------------finalized model----------------')
            self.reg_log = GradientBoostingClassifier()
            self.reg_log.fit(self.training_data_res_transformation, self.y_train_res)
            logger.info(f'Model Test Accuracy: {accuracy_score(self.y_test, self.reg_log.predict(self.testing_data_transformation))}')
            logger.info(f'Confusion Matrix: {confusion_matrix(self.y_test, self.reg_log.predict(self.testing_data_transformation))}')
            logger.info(f'Classification Report: {classification_report(self.y_test, self.reg_log.predict(self.testing_data_transformation))}')

            logger.info(f'--------------Model Saving----------------')
            with open('C:\\Users\\LENOVO\\Documents\\PROJECT_ALL_FILES\\Result\\churn_prediction_project.pkl', 'wb') as f:
                pickle.dump(self.reg_log, f)

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to {er_msg}')


if __name__ == "__main__":
    try:
        path = "C:\\Users\\LENOVO\\Documents\\PROJECT_ALL_FILES\\WA_Fn-UseC_-Telco-Customer-Churn.csv"
        obj = CHURN_PROJECT_INFO(path)
        obj.missing_values()
        obj.handle_outliers()
        obj.feature_selection()
        obj.cat_to_num()
        obj.combining_both_cat_num()
        obj.balanced_data()
        obj.feature_scaling()
        # obj.train_models()
        obj.best_model()
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to {er_msg}')

