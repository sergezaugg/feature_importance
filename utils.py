#--------------------             
# Author : Serge Zaugg
# Description : Utility functions used by main.py
#--------------------

import os
import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
# from utils import make_dataset, fit_rf_get_metrics
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots


def bivariate_normal(n = 1000, mu =[0,0] , std = [3,2], corr = 0.5):
    """ 
    Description: Samples from a bi-variate normal distribution parametrized intuitively with correlation an standard deviation
    Arguments:
    n    : int, the sample size 
    mu   : [float, float], the mean vector
    std  : [float, float], the standard deviations of the the two variables
    corr : float, the correlation between the two variables
    Returns: An n-by-2 numpy array of floats
    """
    mu = np.array(mu)
    std = np.diag(np.array(std))
    sigma1 = np.array([[1.0,corr],[corr,1.0]])
    xtemp = np.matmul(sigma1, std)
    covar1 = np.matmul(std, xtemp)
    x1 = np.random.multivariate_normal(mean = mu, cov = covar1, size=n)
    return(x1)


def make_dataset(params): 
    """  
    Description: Creates a dataset with a binary target variable and 3 continuous predictors (= features)  
    The first and second predictors can be informative for supervised classification, while the third is always non-informative! 
    Arguments:
    params : A dict with 8  key. e.g. {'n1': 100, 'mu1': [0, 0], 'std1': [1, 1], 'corr1': -0.9, 'n2': 100, 'mu2': [0, 0], 'std2': [1, 1], 'corr2': -0.9} 
    Returns: An m-by-4 Pandas dataframe, where m = n1 + n2
    """
    # create mvn data with controlled structure
    x1 = bivariate_normal(n = params['n1'], mu = params['mu1'] , std = params['std1'], corr = params['corr1'])
    x2 = bivariate_normal(n = params['n2'], mu = params['mu2'] , std = params['std2'], corr = params['corr2'])
    # add a class for supervised classification
    x1 = pd.DataFrame(x1, columns = ["f01", "f02"])
    x1["class"] = "Class A"
    x2 = pd.DataFrame(x2, columns = ["f01", "f02"])
    x2["class"] = "Class B"
    df = pd.concat([x1, x2])
    # add a third feature that is  uninformative for classification
    df["f03"] = np.random.normal(0, 1, df.shape[0])
    # re-order columns nicely 
    df = df[['class', 'f01', 'f02', 'f03']]
    return(df)





# devel code - supress execution if this is imported as module 
if __name__ == "__main__":

    xx = bivariate_normal(n = 1000, mu =[1,1] , std = [1,1], corr = -0.9)
    xx.shape
    xx.std(0)
    pd.DataFrame(xx).corr()

    mvn_params = {
        'n1' : 100, 'mu1' : [0,0] , 'std1' : [1,1], 'corr1' : -0.9,
        'n2' : 100, 'mu2' : [0,0] , 'std2' : [1,1], 'corr2' : -0.9,
        }

    df = make_dataset(params = mvn_params) 
    df.head()
    df.shape

    "{:1.2f}".format(456.67895) # check 





# loop over feature sets and fit RFO models
@st.cache_data
def fit_rf_get_metrics(df_data, feat_li, rfo_n_trees = 5, random_seed = 123):
    df_resu = []
    for feat_sel in feat_li:
        X = df_data[feat_sel]
        y = df_data['class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.60)
            # initialize a model for supervised classification 
        clf = RandomForestClassifier(n_estimators=rfo_n_trees, max_depth=30, max_features = 1, random_state=random_seed)
        clf.fit(X_train, y_train)
        # get overall performance as ROC-AUC
        y_pred = clf.predict_proba(X_test)[:,1]
        resu_auc = np.round(roc_auc_score(y_test, y_pred),2).item()
        resu_auc = "{:1.2f}".format(resu_auc)
        # get gini-based feature importance
        resu_imp = (clf.feature_importances_).round(2).tolist()
        resu_imp = ["{:1.2f}".format(a) for a  in resu_imp]
        # prepare results to be organized in a data frame
        col_values = [[resu_auc] + resu_imp]
        col_names = ['Importance_' + a for a in feat_sel]
        col_names = ['AUC_Test'] + col_names
        df_t = pd.DataFrame(col_values, columns = col_names)
        # append meta-data on the right of df 
        incl_features_str = ", ".join(feat_sel)
        df_t['Included_Features'] = incl_features_str
        # store in list 
        df_resu.append(df_t)
    df_resu = pd.concat(df_resu)
    return(df_resu)
