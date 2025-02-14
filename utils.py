#--------------------             
# Author : Serge Zaugg
# Description : 
#--------------------

import os
import numpy as np
import pandas as pd

def bivariate_normal(n = 1000, mu =[0,0] , std = [3,2], corr = 0.5):
    """   """
    mu = np.array(mu)
    std = np.diag(np.array(std))
    sigma1 = np.array([[1.0,corr],[corr,1.0]])
    xtemp = np.matmul(sigma1, std)
    covar1 = np.matmul(std, xtemp)
    x1 = np.random.multivariate_normal(mean = mu, cov = covar1, size=n)
    return(x1)


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





# devel code - supress execution if this is imported as module 
if __name__ == "__main__":

    xx = bivariate_normal(n = 1000, mu =[1,1] , std = [1,1], corr = -0.9)
    xx.std(0)
    pd.DataFrame(xx).corr()


    mvn_params = {
        'n1' : 100, 'mu1' : [0,0] , 'std1' : [1,1], 'corr1' : -0.9,
        'n2' : 100, 'mu2' : [0,0] , 'std2' : [1,1], 'corr2' : -0.9,
        }

    df = make_dataset(params = mvn_params) 
    df.head()
