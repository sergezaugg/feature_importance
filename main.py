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

# separable but only if f1 and f1 included 
mvn_params = {
    'n1' : 5000, 'mu1' : [0,0] , 'std1' : [1,1], 'corr1' : -0.9,
    'n2' : 5000, 'mu2' : [0,0] , 'std2' : [1,1], 'corr2' : +0.9
    }

# f1 alon is totally uninformative, 
mvn_params = {
    'n1' : 5000, 'mu1' : [0,0] , 'std1' : [1,1], 'corr1' : -0.9,
    'n2' : 5000, 'mu2' : [0,3] , 'std2' : [1,1], 'corr2' : -0.9
    }

# not separable 
mvn_params = {
    'n1' : 5000, 'mu1' : [0,0] , 'std1' : [1,1], 'corr1' : -0.9,
    'n2' : 5000, 'mu2' : [0,0] , 'std2' : [1,1], 'corr2' : -0.9,
    }

df = make_dataset(params = mvn_params) 
df.head()

fig = px.scatter(
    data_frame = df,
    x = 'f01',
    y = 'f02',
    # y = 'f03',
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



# include informative features  only 
X = df[["f01", "f02"]]
y = df['class']

# include all features         
X = df[["f01", "f02", "f03"]]
y = df['class']


X = df[["f01", "f03"]]
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.60, random_state=666)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)[:,1]
roc_auc_score(y_test, y_pred)

clf.feature_importances_




