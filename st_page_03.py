#--------------------             
# Author : Serge Zaugg
# Description : 
#--------------------

import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from utils import make_dataset, fit_rf_get_metrics
from sklearn.utils import shuffle
from streamlit import session_state as ss

init_vals = {
    'mean1x' : 0.0,
    'mean1y' : 0.0,
    'stdv1x' : 1.0,
    'stdv1y' : 1.0,
    'corr1'  : 0.0,
    'mean2x' : 0.0,
    'mean2y' : 0.0,
    'stdv2x' : 1.0,
    'stdv2y' : 1.0,
    'corr2'  : 0.0,
    }  

#--------------------------------
# streamlit start here 

# initialize session state 
if 'rfo_n_trees' not in ss:
    ss['rfo_n_trees'] = 30
if 'max_features' not in ss:
    ss['max_features'] = 1
if 'max_depth' not in ss:
    ss['max_depth'] = 10
if 'random_seed' not in ss:
    ss['random_seed'] = 503
if 'distr' not in ss:
    ss['distr'] = init_vals

# reset 
with st.form("reset_01", border=False):
    submitted = st.form_submit_button("Reset")
    if submitted: 
        ss['distr'] = init_vals

#----------------
# 1st line 
col_a, col_b, col_c= st.columns([0.10, 0.10, 0.50, ])

with col_a:
    st.subheader("Class A")
    with st.form("rand_1", border=False):
        submitted = st.form_submit_button("Randomize")
        if submitted: 
            ss['distr']['mean1x'] = np.random.uniform(low=-5.0,  high=+5.0, size=1)[0]
            ss['distr']['mean1y'] = np.random.uniform(low=-5.0,  high=+5.0, size=1)[0]
            ss['distr']['stdv1x'] = np.random.uniform(low= 0.01, high=5.0, size=1)[0]
            ss['distr']['stdv1y'] = np.random.uniform(low=-0.01, high=5.0, size=1)[0]
            ss['distr']['corr1']  = np.random.uniform(low=-1.0,  high=+1.0, size=1)[0]
    numb1  = st.slider("N",       min_value=  10,   max_value=1000, value=300,  key="slide_n1")
    mean1x = st.slider("mean x",  min_value= -5.0,  max_value=+5.0, value=ss['distr']['mean1x'], key="slide_mu1x")
    mean1y = st.slider("mean y",  min_value= -5.0,  max_value=+5.0, value=ss['distr']['mean1y'], key="slide_mu1y")
    stdv1x = st.slider("stdev x", min_value= +0.01, max_value=10.0, value=ss['distr']['stdv1x'], key="slide_std1x")
    stdv1y = st.slider("stdev y", min_value= +0.01, max_value=10.0, value=ss['distr']['stdv1y'], key="slide_std1y")
    corr1  = st.slider("corr",    min_value=-1.0,   max_value=+1.0, value=ss['distr']['corr1'] , key="slide_corr1")
with col_b:   
    st.subheader("Class B")
    with st.form("rand_2", border=False):
        submitted = st.form_submit_button("Randomize")
        if submitted: 
            ss['distr']['mean2x'] =  np.random.uniform(low=-5.0,  high=+5.0, size=1)[0]
            ss['distr']['mean2y'] =  np.random.uniform(low=-5.0,  high=+5.0, size=1)[0]
            ss['distr']['stdv2x'] =  np.random.uniform(low= 0.01, high=5.0, size=1)[0]
            ss['distr']['stdv2y'] =  np.random.uniform(low=-0.01, high=5.0, size=1)[0]
            ss['distr']['corr2']  =  np.random.uniform(low=-1.0,  high=+1.0, size=1)[0]
    numb2  = st.slider("N",       min_value=  10,   max_value=1000, value=300,  key="slide_n2")
    mean2x = st.slider("mean x",  min_value= -5.0,  max_value=+5.0, value=ss['distr']['mean2x'], key="slide_mu2x")
    mean2y = st.slider("mean y",  min_value= -5.0,  max_value=+5.0, value=ss['distr']['mean2y'], key="slide_mu2y")
    stdv2x = st.slider("stdev x", min_value= +0.01, max_value=10.0, value=ss['distr']['stdv2x'], key="slide_std2x")
    stdv2y = st.slider("stdev y", min_value= +0.01, max_value=10.0, value=ss['distr']['stdv2y'], key="slide_std2y")
    corr2  = st.slider("corr",    min_value=-1.0,   max_value=+1.0, value=ss['distr']['corr2'] , key="slide_corr2")


#----------------
# computation block 
scenario_di = { 
        'n1' : numb1, 'mu1' : [mean1x, mean1y] , 'std1' : [stdv1x, stdv1y], 'corr1' : corr1,
        'n2' : numb2, 'mu2' : [mean2x, mean2y] , 'std2' : [stdv2x, stdv2y], 'corr2' : corr2,
        }
df_data = make_dataset(params = scenario_di) 
df_data = shuffle(df_data)
df_data = df_data.sort_values(by='class')

fig1 = px.scatter_3d(
    data_frame = df_data,
    x = 'f01',
    y = 'f02',
    z = 'f03',
    color = 'class',
    width = 600,
    height = 600,
    title = "",
    color_discrete_sequence=['#22ff99', '#9911ff']
    )


_ = fig1.update_xaxes(showline = True, linecolor = 'white', linewidth = 1, row = 1, col = 1, mirror = True)
_ = fig1.update_yaxes(showline = True, linecolor = 'white', linewidth = 1, row = 1, col = 1, mirror = True)
_ = fig1.update_layout(paper_bgcolor="#112233",)
# _ = fig1.update(layout_xaxis_range = [-15,+15])
# _ = fig1.update(layout_yaxis_range = [-15,+15])
fig1.update_traces(marker=dict(size=5))
#----------------


with col_c:
    st.subheader("Scatterplot of scenario")
    st.plotly_chart(fig1, use_container_width=False)

