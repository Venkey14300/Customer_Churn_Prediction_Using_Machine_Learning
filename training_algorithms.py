import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import logging
from log_code import setup_logging
logger = setup_logging('train_algo')
import warnings
warnings.filterwarnings("ignore")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import roc_curve


def knn_algorithm(X_train,y_train, X_test, y_test):
    try:
        knn_reg = KNeighborsClassifier(n_neighbors=5)
        knn_reg.fit(X_train, y_train)
        logger.info(f'Test Accuracy KNN : {accuracy_score(y_test, knn_reg.predict(X_test))}')
        logger.info(f'Confusion Matrix KNN : {confusion_matrix(y_test, knn_reg.predict(X_test))}')
        logger.info(f'Classification Report KNN : {classification_report(y_test, knn_reg.predict(X_test))}')
        global knn_pred
        knn_pred = knn_reg.predict(X_test)

    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to {er_msg}')

def Decision_tree_algorithm(X_train,y_train, X_test, y_test):
    try:
        DT_reg = DecisionTreeClassifier(criterion='entropy')
        DT_reg.fit(X_train, y_train)
        logger.info(f'Test Accuracy DT : {accuracy_score(y_test, DT_reg.predict(X_test))}')
        logger.info(f'Confusion Matrix DT : {confusion_matrix(y_test, DT_reg.predict(X_test))}')
        logger.info(f'Classification Report DT : {classification_report(y_test, DT_reg.predict(X_test))}')
        global DT_pred
        DT_pred = DT_reg.predict(X_test)

    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to {er_msg}')

def Random_Forest_algorithm(X_train,y_train, X_test, y_test):
    try:
        RF_reg = RandomForestClassifier(criterion='entropy', n_estimators=5)
        RF_reg.fit(X_train, y_train)
        logger.info(f'Test Accuracy RF : {accuracy_score(y_test, RF_reg.predict(X_test))}')
        logger.info(f'Confusion Matrix RF : {confusion_matrix(y_test, RF_reg.predict(X_test))}')
        logger.info(f'Classification Report RF : {classification_report(y_test, RF_reg.predict(X_test))}')
        global RF_pred
        RF_pred = RF_reg.predict(X_test)

    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to {er_msg}')

def Gradient_Boosting_algorithm(X_train,y_train, X_test, y_test):
    try:
        GB_reg = GradientBoostingClassifier()
        GB_reg.fit(X_train, y_train)
        logger.info(f'Test Accuracy GB : {accuracy_score(y_test, GB_reg.predict(X_test))}')
        logger.info(f'Confusion Matrix GB : {confusion_matrix(y_test, GB_reg.predict(X_test))}')
        logger.info(f'Classification Report GB : {classification_report(y_test, GB_reg.predict(X_test))}')
        global GB_pred
        GB_pred = GB_reg.predict(X_test)

    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to {er_msg}')

def SVM_algorithm(X_train,y_train, X_test, y_test):
    try:
        SVM_reg = SVC(kernel='rbf')
        SVM_reg.fit(X_train, y_train)
        logger.info(f'Test Accuracy SVM : {accuracy_score(y_test, SVM_reg.predict(X_test))}')
        logger.info(f'Confusion Matrix SVM : {confusion_matrix(y_test, SVM_reg.predict(X_test))}')
        logger.info(f'Classification Report SVM : {classification_report(y_test, SVM_reg.predict(X_test))}')
        global SVM_pred
        SVM_pred = SVM_reg.predict(X_test)

    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to {er_msg}')

def Navie_bayes_algorithm(X_train,y_train, X_test, y_test):
    try:
        NB_reg = GaussianNB()
        NB_reg.fit(X_train, y_train)
        logger.info(f'Test Accuracy NB : {accuracy_score(y_test, NB_reg.predict(X_test))}')
        logger.info(f'Confusion Matrix NB : {confusion_matrix(y_test, NB_reg.predict(X_test))}')
        logger.info(f'Classification Report NB : {classification_report(y_test, NB_reg.predict(X_test))}')
        global NB_pred
        NB_pred = NB_reg.predict(X_test)

    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to {er_msg}')

def Logistic_Regression_algorithm(X_train,y_train, X_test, y_test):
    try:
        lg_reg = LogisticRegression()
        lg_reg.fit(X_train, y_train)
        logger.info(f'Test Accuracy LR : {accuracy_score(y_test, lg_reg.predict(X_test))}')
        logger.info(f'Confusion Matrix LR : {confusion_matrix(y_test, lg_reg.predict(X_test))}')
        logger.info(f'Classification Report LR : {classification_report(y_test, lg_reg.predict(X_test))}')
        global LR_pred
        LR_pred = lg_reg.predict(X_test)

    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to {er_msg}')

def XGB_algorithm(X_train,y_train, X_test, y_test):
    try:
        XGB_reg = XGBClassifier()
        XGB_reg.fit(X_train, y_train)
        logger.info(f'Test Accuracy KNN : {accuracy_score(y_test, XGB_reg.predict(X_test))}')
        logger.info(f'Confusion Matrix : {confusion_matrix(y_test, XGB_reg.predict(X_test))}')
        logger.info(f'Classification Report : {classification_report(y_test, XGB_reg.predict(X_test))}')
        global XGB_pred
        XGB_pred = XGB_reg.predict(X_test)

    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to {er_msg}')


def best_model_using_auc_roc(X_train, y_train, X_test, y_test):
    try:
        knn_fpr, knn_tpr, knn_thre = roc_curve(y_test, knn_pred)
        nb_fpr, nb_tpr, nb_thre = roc_curve(y_test, NB_pred)
        lr_fpr, lr_tpr, lr_thre = roc_curve(y_test, LR_pred)
        dt_fpr, dt_tpr, dt_thre = roc_curve(y_test, DT_pred)
        rf_fpr, rf_tpr, rf_thre = roc_curve(y_test, RF_pred)
        gb_fpr, gb_tpr, gb_thre = roc_curve(y_test, GB_pred)
        xgb_fpr, xgb_tpr, xgb_thre = roc_curve(y_test, XGB_pred)
        svm_fpr, svm_tpr, svm_thre = roc_curve(y_test, SVM_pred)

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for all algorithms')
        plt.plot([0,1],[0,1],'k--')
        plt.plot(knn_fpr, knn_tpr, color='r', label='KNN Algorithm')
        plt.plot(nb_fpr, nb_tpr, color='blue', label='NB Algorithm')
        plt.plot(lr_fpr, lr_tpr, color='green', label='LG Algorithm')
        plt.plot(dt_fpr, dt_tpr, color='yellow', label='DT Algorithm')
        plt.plot(rf_fpr, rf_tpr, color='pink', label='RF Algorithm')
        plt.plot(gb_fpr, gb_tpr, color='black', label='GB Algorithm')
        plt.plot(xgb_fpr, xgb_tpr, color='orange', label='XGB Algorithm')
        plt.plot(svm_fpr, svm_tpr, color='brown', label='SVM Algorithm')
        plt.legend(loc=0)
        plt.show()

    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to {er_msg}')


def common(X_train,y_train, X_test, y_test):
    try:
        logger.info(f'Giving Data to every Function')
        logger.info(f'--------knn---------')
        knn_algorithm(X_train,y_train, X_test, y_test)

        logger.info(f'--------DT---------')
        Decision_tree_algorithm(X_train, y_train, X_test, y_test)

        logger.info(f'--------RF---------')
        Random_Forest_algorithm(X_train, y_train, X_test, y_test)

        logger.info(f'--------GB---------')
        Gradient_Boosting_algorithm(X_train, y_train, X_test, y_test)

        logger.info(f'--------SVM---------')
        SVM_algorithm(X_train, y_train, X_test, y_test)

        logger.info(f'--------NB---------')
        Navie_bayes_algorithm(X_train, y_train, X_test, y_test)

        logger.info(f'--------LR---------')
        Logistic_Regression_algorithm(X_train, y_train, X_test, y_test)

        logger.info(f'--------XGB---------')
        XGB_algorithm(X_train, y_train, X_test, y_test)

        logger.info(f'--------AUC ROC---------')
        best_model_using_auc_roc(X_train, y_train, X_test, y_test)

    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to {er_msg}')