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
import plotly.graph_objects as go
from plotly.subplots import make_subplots


#-----------------------------------------------------
# (1 ) choose a scenario (the know truth)

scenarios_di = { 
    "Features are NOT informative" : {
        'n1' : 5000, 'mu1' : [0,0] , 'std1' : [1,1], 'corr1' : -0.9,
        'n2' : 5000, 'mu2' : [0,0] , 'std2' : [1,1], 'corr2' : -0.9,
        }
    , 
    "Both features are informative" : {
        'n1' : 5000, 'mu1' : [ 1, 1] , 'std1' : [1,1], 'corr1' : 0.7,
        'n2' : 5000, 'mu2' : [-1,-1] , 'std2' : [1,1], 'corr2' : 0.7
        }
    ,
    "only one features is informative " : {
        'n1' : 5000, 'mu1' : [ 1, 1] , 'std1' : [1,1], 'corr1' : 0.0,
        'n2' : 5000, 'mu2' : [-1, 1] , 'std2' : [1,1], 'corr2' : 0.0
        }
    ,
    "Information in interaction - parallel" : {
        'n1' : 5000, 'mu1' : [-0.2,-0.2] , 'std1' : [1,1], 'corr1' : -0.98,
        'n2' : 5000, 'mu2' : [+0.2,+0.2] , 'std2' : [1,1], 'corr2' : -0.98
        }
    ,
    "Information in interaction - cross" : {
        'n1' : 5000, 'mu1' : [0,0] , 'std1' : [1,1], 'corr1' : -0.98,
        'n2' : 5000, 'mu2' : [0,0] , 'std2' : [1,1], 'corr2' : +0.98
        }
    ,
    }


# loop over scenarios 
for k in scenarios_di:
    print(k)
    mvn_params = scenarios_di[k]


    #-----------------------------------------------------
    # (2) generate the dataset and plot it 
    df = make_dataset(params = mvn_params) 
    # df.head()

    fig1 = px.scatter(
        data_frame = df,
        x = 'f01',
        y = 'f02',
        color = 'class',
        width = 500,
        height = 500,
        title = k
        )
    # fig1.show()



    #-----------------------------------------------------
    # (3) fit models for supervised classification and store performance and feature importance metrics
    clf = RandomForestClassifier(
        n_estimators=100,               
        max_depth=20, 
        random_state=0, 
        max_features = 1)

    # 
    feat_li = [
        ["f01", "f02", "f03"],
        ["f01", "f03"],
        ["f02", "f03"],
        ["f01", "f02"],
        ]

    # loop over feature sets 
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
        resu_imp = (clf.feature_importances_).round(2).tolist()
        # prepare results to be organized in a data frame
        col_values = [[resu_auc] + resu_imp]
        col_names = ['Importance_' + a for a in feat_sel]
        col_names = ['AUC_Test'] + col_names
        df_t = pd.DataFrame(col_values, columns = col_names)
        # append meta-data on teh right of df 
        incl_features_str = ", ".join(feat_sel)
        df_t['Included_Features'] = incl_features_str
        # store in list 
        df_resu.append(df_t)
    df_resu = pd.concat(df_resu)


    # show table in a plotly figure 
    fig2 = go.Figure(data=[go.Table(
        header=dict(values=list(df_resu.columns),
                    fill_color='black',
                    align='left'),
        cells=dict(values=[df_resu.AUC_Test, df_resu.Importance_f01, df_resu.Importance_f02, df_resu.Importance_f03, df_resu.Included_Features],
                fill_color='black',
                align='left'))
    ])

    # combine subplots to the final plot 
    # the specs argument is a ****** pain in the ***, I still love plotly
    fig = make_subplots(rows=2, cols=1,  specs=[[{'type': 'xy'}], [{'type': 'table'}]] , row_heights =[0.8, 0.2]  )
    fig.add_trace(fig1['data'][0], row=1, col=1)
    fig.add_trace(fig1['data'][1], row=1, col=1)
    fig.add_trace(fig2['data'][0], row=2, col=1)

    _ = fig.update_layout(template="plotly_dark")
    _ = fig.update_layout(autosize=False,width=700,height=900,)
    _ = fig.update_layout(title_text=k,title_font_size=25)
    fig.show()








