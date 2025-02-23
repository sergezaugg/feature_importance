#--------------------             
# Author : Serge Zaugg
# Description : Create synthetic datasets for classification, train models, assess classification performance and feature importance
#--------------------

import numpy as np
import streamlit as st
import plotly.express as px
from utils import make_dataset, fit_rf_get_metrics
import plotly.graph_objects as go

# streamlit run stmain.py

random_seed = 557
np.random.seed(seed=random_seed)

feat_li = [
    ["f01", "f02", "f03"],
    ["f01", "f03"],
    ["f02", "f03"],
    ["f01", "f02"],
    ["f03"],
    ]

#--------------------------------
# streamlit start here 

# st.set_page_config(layout="wide")

col_aa, col_bb, = st.columns([0.85, 0.15])

with col_aa: 
    st.title('Can we really rank features according to their importance?')
  

with col_bb:
    st.text("more")
    rfo_n_trees = st.number_input("N trees random forest", min_value=10, max_value=100, value=10, step=10)
   

col_a, col_b, col_space01, col_c, col_d, = st.columns([0.10, 0.10, 0.05, 0.50, 0.5])

with col_a:
    st.subheader("Class A")
    numb1 = st.slider("N",     min_value=  10,   max_value=1000,    value=100,      key="slide_n1")
    mean1x = st.slider("mean x",  min_value= -5.0,  max_value=+5.0, value=1.0,  key="slide_mu1x")
    mean1y = st.slider("mean y",  min_value= -5.0,  max_value=+5.0, value=1.0,  key="slide_mu1y")
    stdv1x = st.slider("stdev x", min_value= +0.01, max_value=10.0, value=1.0, key="slide_std1x")
    stdv1y = st.slider("stdev y", min_value= +0.01, max_value=10.0, value=1.0, key="slide_std1y")
    corr1 = st.slider("corr",  min_value=-1.0, max_value=+1.0,      value=0.9 ,  key="slide_corr1")
with col_b:   
    st.subheader("Class B")
    numb2 = st.slider("N",     min_value=  10,   max_value=1000,    value=100,      key="slide_n2")
    mean2x = st.slider("mean x",  min_value= -5.0,  max_value=+5.0, value=1.0,  key="slide_mu2x")
    mean2y = st.slider("mean y",  min_value= -5.0,  max_value=+5.0, value=1.0,  key="slide_mu2y")
    stdv2x = st.slider("stdev x", min_value= +0.01, max_value=10.0, value=1.0, key="slide_std2x")
    stdv2y = st.slider("stdev y", min_value= +0.01, max_value=10.0, value=1.0, key="slide_std2y")
    corr2 = st.slider("corr",  min_value=-1.0, max_value=+1.0,      value=-0.9,   key="slide_corr2")

scenario_di = { 
        'n1' : numb1, 'mu1' : [mean1x, mean1y] , 'std1' : [stdv1x, stdv1y], 'corr1' : corr1,
        'n2' : numb2, 'mu2' : [mean2x, mean2y] , 'std2' : [stdv2x, stdv2y], 'corr2' : corr2,
        }

df_data = make_dataset(params = scenario_di) 

df_resu = fit_rf_get_metrics(df_data, feat_li, rfo_n_trees = rfo_n_trees, random_seed = random_seed, max_features = 1, max_depth = 30)

fig1 = px.scatter(
    data_frame = df_data,
    x = 'f01',
    y = 'f02',
    color = 'class',
    width = 600,
    height = 600,
    title = "",
    color_discrete_sequence=['#ee33ff', '#33aaff']
    )
_ = fig1.update_xaxes(showline = True, linecolor = 'white', linewidth = 1, row = 1, col = 1, mirror = True)
_ = fig1.update_yaxes(showline = True, linecolor = 'white', linewidth = 1, row = 1, col = 1, mirror = True)
_ = fig1.update_layout(paper_bgcolor="#112233",)
fig1.update_traces(marker=dict(size=5))


with col_c:
    st.subheader("Scatterplot of scenario")
    st.plotly_chart(fig1, use_container_width=False)
with col_d:
    st.subheader("Predictive performance and feature importance")
    st.dataframe(df_resu, hide_index=True)  

