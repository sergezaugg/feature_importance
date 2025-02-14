#--------------------             
# Author : Serge Zaugg
# Description : 
#--------------------

import os
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from utils import make_dataset




# scenario 1 - parallel
mvn_params = {
    'n1' : 5000, 'mu1' : [-0.2,-0.2] , 'std1' : [1,1], 'corr1' : -0.98,
    'n2' : 5000, 'mu2' : [+0.2,+0.2] , 'std2' : [1,1], 'corr2' : -0.98
    }


# scenario 2 - cross
mvn_params = {
    'n1' : 5000, 'mu1' : [0,0] , 'std1' : [1,1], 'corr1' : -0.9,
    'n2' : 5000, 'mu2' : [0,0] , 'std2' : [1,1], 'corr2' : +0.9
    }


# not separable 
mvn_params = {
    'n1' : 5000, 'mu1' : [0,0] , 'std1' : [1,1], 'corr1' : -0.9,
    'n2' : 5000, 'mu2' : [0,0] , 'std2' : [1,1], 'corr2' : -0.9,
    }


    
# scenario 0 - easy blobs
mvn_params = {
    'n1' : 5000, 'mu1' : [ 1, 1] , 'std1' : [1,1], 'corr1' : 0.7,
    'n2' : 5000, 'mu2' : [-1,-1] , 'std2' : [1,1], 'corr2' : 0.7
    }

df = make_dataset(params = mvn_params) 
df.head()

fig = px.scatter(
    data_frame = df,
    x = 'f01',
    y = 'f02',
    color = 'class',
    width = 500,
    height = 500,
    )
fig.update_layout(template="plotly_dark")
fig.show()



clf = RandomForestClassifier(
    n_estimators=100,               
    max_depth=20, 
    random_state=0, 
    max_features = 3)


# 
feat_li = [
    ["f01", "f02", "f03"],
    ["f01", "f03"],
    ["f02", "f03"],
    ]

df_resu = []
for feat_sel in feat_li:
    # fit RFO model
    X = df[feat_sel]
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.60)
    clf.fit(X_train, y_train)
    # get overall performance as ROC-AUC
    y_pred = clf.predict_proba(X_test)[:,1]
    resu_auc = np.round(roc_auc_score(y_test, y_pred),2).item()
    # get gini-based feature importance
    resu_imp = (clf.feature_importances_).round(3).tolist()

    # prepare results to be organized in a data frame
    col_values = [[resu_auc] + resu_imp]
    col_names = ['Importance_' + a for a in feat_sel]
    col_names = ['AUC_Test'] + col_names
    df_resu.append(pd.DataFrame(col_values, columns = col_names))

pd.concat(df_resu)





