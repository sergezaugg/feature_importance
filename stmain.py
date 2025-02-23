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
from utils import make_dataset, fit_rf_get_metrics
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# streamlit run stmain.py

random_seed = 557
np.random.seed(seed=random_seed)

rfo_n_trees = 10


feat_li = [
    ["f01", "f02", "f03"],
    ["f01", "f03"],
    ["f02", "f03"],
    ["f01", "f02"],
    ["f03"],
    ]


#--------------------------------
# streamlit start here 

st.set_page_config(layout="wide")

st.title('Feature importance')


with st.sidebar:
    st.text("aaaa")

    # Define several scenarios 
    col0, col1, col2, col3, col4, col5, col6 = st.columns(7*[1,])
    with col0:
        st.text("Class 1")
    with col1:
        numb1 = st.slider("N",     min_value=  10,   max_value=1000, value=100, key="slide_n1")
    with col2:
        mean1x = st.slider("mean x",  min_value= -5.0,  max_value=+5.0, value=25.0, key="slide_mu1x")
    with col3:
        mean1y = st.slider("mean y",  min_value= -5.0,  max_value=+5.0, value=25.0, key="slide_mu1y")
    with col4:
        stdv1x = st.slider("stdev x", min_value= +0.01, max_value=10.0, value=1.0, key="slide_std1x")
    with col5:
        stdv1y = st.slider("stdev y", min_value= +0.01, max_value=10.0, value=1.0, key="slide_std1y")
    with col6:
        corr1 = st.slider("corr",  min_value=-1.0, max_value=+1.0, value=0.2, key="slide_corr1")

        
    col0, col1, col2, col3, col4, col5, col6 = st.columns(7*[1,])
    with col0:
        st.text("Class 2")
    with col1:
        numb2 = st.slider("N",     min_value=  10,   max_value=1000, value=100, key="slide_n2")
    with col2:
        mean2x = st.slider("mean x",  min_value= -5.0,  max_value=+5.0, value=25.0, key="slide_mu2x")
    with col3:
        mean2y = st.slider("mean y",  min_value= -5.0,  max_value=+5.0, value=25.0, key="slide_mu2y")
    with col4:
        stdv2x = st.slider("stdev x", min_value= +0.01, max_value=10.0, value=1.0, key="slide_std2x")
    with col5:
        stdv2y = st.slider("stdev y", min_value= +0.01, max_value=10.0, value=1.0, key="slide_std2y")
    with col6:
        corr2 = st.slider("corr",  min_value=-1.0, max_value=+1.0, value=0.2, key="slide_corr2")


scenario_di = { 
        'n1' : numb1, 'mu1' : [mean1x, mean1y] , 'std1' : [stdv1x, stdv1y], 'corr1' : corr1,
        'n2' : numb2, 'mu2' : [mean2x, mean2y] , 'std2' : [stdv2x, stdv2y], 'corr2' : corr2,
        }

# st.write("Values:", scenario_di)

df_data = make_dataset(params = scenario_di) 

df_resu = fit_rf_get_metrics(df_data, feat_li, rfo_n_trees = rfo_n_trees, random_seed = random_seed)

fig1 = px.scatter(
    data_frame = df_data,
    x = 'f01',
    y = 'f02',
    color = 'class',
    width = 600,
    height = 600,
    title = "aaa"
    )
fig1.update_traces(marker=dict(size=5))


# 
col0, col1 = st.columns(2*[1,])
with col0:
    st.plotly_chart(fig1, use_container_width=False)
with col1:
    st.dataframe(df_resu)  

