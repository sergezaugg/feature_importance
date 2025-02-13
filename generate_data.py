#--------------------             
#
#--------------------

import os
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def bivariate_normal(n = 1000, mu =[0,0] , std = [3,2], corr = 0.5):
    """   """
    mu = np.array(mu)
    std = np.diag(np.array(std))
    sigma1 = np.array([[1.0,corr],[corr,1.0]])
    xtemp = np.matmul(sigma1, std)
    covar1 = np.matmul(std, xtemp)
    x1 = np.random.multivariate_normal(mean = mu, cov = covar1, size=n)
    return(x1)

# check
xx = bivariate_normal(n = 1000, mu =[1,1] , std = [1,1], corr = -0.9)
xx.std(0)
pd.DataFrame(xx).corr()


def make_dataset(params): 
    """    """
    # create mvn data with controlled structure
    x1 = bivariate_normal(n = params['n1'], mu = params['mu1'] , std = params['std1'], corr = params['corr1'])
    x2 = bivariate_normal(n = params['n2'], mu = params['mu2'] , std = params['std2'], corr = params['corr2'])
    # add a class for supervied classification
    x1 = pd.DataFrame(x1, columns = ["f01", "f02"])
    x1["class"] = "a"
    x2 = pd.DataFrame(x2, columns = ["f01", "f02"])
    x2["class"] = "b"
    df = pd.concat([x1, x2])
    # add a thrd feateure that is  uninformative for classification
    df["f03"] = np.random.normal(0, 1, df.shape[0])
    # re-order columns nicely 
    df = df[['class', 'f01', 'f02', 'f03']]
    return(df)


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
