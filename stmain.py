#--------------------             
# Author : Serge Zaugg
# Description : Create synthetic datasets for classification, train models, assess classification performance and feature importance
#--------------------

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from utils import make_dataset
import plotly.graph_objects as go
from plotly.subplots import make_subplots


rfo_n_trees = 5

# streamlit run stmain.py

random_seed = 557
np.random.seed(seed=random_seed)



st.title('Feature importance')


#-----------------------------
# (1) Define several scenarios 

col1, col2, col3, col4, col5, col6 = st.columns([1,1,1,1,1,1])
with col1:
    numb1 = st.slider("N 1",     min_value=  10,   max_value=1000, value=100, key="slide_n1")
with col2:
    mean1x = st.slider("mean 1x",  min_value= -5.0,  max_value=+5.0, value=25.0, key="slide_mu1x")
with col3:
    mean1y = st.slider("mean 1y",  min_value= -5.0,  max_value=+5.0, value=25.0, key="slide_mu1y")
with col4:
    stdv1x = st.slider("stdev 1x", min_value= +0.01, max_value=10.0, value=1.0, key="slide_std1x")
with col5:
    stdv1y = st.slider("stdev 1y", min_value= +0.01, max_value=10.0, value=1.0, key="slide_std1y")
with col6:
    corr1 = st.slider("corr 1",  min_value=-1.0, max_value=-1.0, value=0.2, key="slide_corr1")

    


scenarios_di = { 
    "Scenario custom" : {
        'n1' : numb1, 'mu1' : [mean1x, mean1y] , 'std1' : [stdv1x, stdv1y], 'corr1' : corr1,
        'n2' : 10, 'mu2' : [1,1] , 'std2' : [1,1], 'corr2' : 0.00,
        }}


st.write("Values:", scenarios_di)




#--------------------------------
# (2) define several feature sets
feat_li = [
    ["f01", "f02", "f03"],
    ["f01", "f03"],
    # ["f02", "f03"],
    # ["f01", "f02"],
    # ["f03"],
    ]


#-----------------------------------------
# (3) loop over scenarios and feature sets
for k in scenarios_di:
    print(k)
    mvn_params = scenarios_di[k]

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
    fig1.update_traces(marker=dict(size=5))
    # fig1.show()

    # loop over feature sets and fit RFO models
    df_resu = []
    for feat_sel in feat_li:
        X = df[feat_sel]
        y = df['class']
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
    fig = make_subplots(rows=2, cols=1,  specs=[[{'type': 'xy'}], [{'type': 'table'}]] , row_heights =[0.8, 0.2]  )
    fig.add_trace(fig1['data'][0], row=1, col=1)
    fig.add_trace(fig1['data'][1], row=1, col=1)
    fig.add_trace(fig2['data'][0], row=2, col=1)
    fig['layout']['xaxis']['title']='f01'
    fig['layout']['yaxis']['title']='f02'
    _ = fig.update_layout(template="plotly_dark")
    _ = fig.update_layout(autosize=False,width=750,height=950,)
    _ = fig.update_layout(title_text=k,title_font_size=15)
    # fig.show()

    st.plotly_chart(fig)



