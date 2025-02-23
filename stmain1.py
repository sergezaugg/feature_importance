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

random_seed = 554 # 554
np.random.seed(seed=random_seed)

feat_li = [
    ["f01", "f02", "f03"],
    ["f01", "f03"],
    ["f02", "f03"],
    # ["f01", "f02"],
    # ["f03"],
    ]

#--------------------------------
# streamlit start here 

# st.set_page_config(layout="wide")

col_aa, col_bb, = st.columns([0.85, 0.15])

with col_aa: 
    st.title('Can we really rank features according to their importance?')
   
with col_bb:
    st.text("more")
    rfo_n_trees = st.number_input("N trees random forest", min_value=10, max_value=100, value=30, step=10)
    max_features = st.number_input("max features", min_value=1, max_value=3, value=1, step=1)
   
col_a, col_space01, col_c, col_d, = st.columns([0.20, 0.05, 0.50, 0.5])

preset_options = [
    "Both feat. informative",
    "Both feat. informative+redundant",
    "Joint information feat. - parallel",
    "Joint information feat. - cross",
    # "Feat. not informative",
    # "One feat. informative",
    ]

with col_a:
    option1 = st.selectbox("Select", preset_options, key = 'sel02')

if option1 == preset_options[0]:
    scenario_di = {
        'n1' : 10000, 'mu1' : [0.0, 2.0] , 'std1' : [1.0,1.0], 'corr1' : 0.00,
        'n2' : 10000, 'mu2' : [2.0, 0.0] , 'std2' : [1.0,1.0], 'corr2' : 0.00,
        }
if option1 == preset_options[1]:
    scenario_di = {
        'n1' : 10000, 'mu1' : [ 1.4,  1.4] , 'std1' : [1.0,1.0], 'corr1' : +0.98,
        'n2' : 10000, 'mu2' : [-1.4, -1.4] , 'std2' : [1.0,1.0], 'corr2' : +0.98,
        }
if option1 == preset_options[2]:
    scenario_di = {
        'n1' : 10000, 'mu1' : [-0.14, -0.14] , 'std1' : [1.0,1.0], 'corr1' : -0.98,
        'n2' : 10000, 'mu2' : [+0.14, +0.14] , 'std2' : [1.0,1.0], 'corr2' : -0.98,
        }
if option1 == preset_options[3]:
    scenario_di = {
        'n1' : 10000, 'mu1' : [0.0, 0.0] , 'std1' : [1.1,1.1], 'corr1' : -0.98,
        'n2' : 10000, 'mu2' : [0.0, 0.0] , 'std2' : [1.1,1.1], 'corr2' : +0.98,
        }
# if option1 == preset_options[4]:
#     scenario_di ={
#         'n1' : 10000, 'mu1' : [0.0, 0.0] , 'std1' : [1,1], 'corr1' : -0.90,
#         'n2' : 10000, 'mu2' : [0.0, 0.0] , 'std2' : [1,1], 'corr2' : -0.90,
#         }
# if option1 == preset_options[5]:
#     scenario_di ={
#         'n1' : 10000, 'mu1' : [ 1.0, 1.0] , 'std1' : [1,1], 'corr1' : 0.00,
#         'n2' : 10000, 'mu2' : [-1.0, 1.0] , 'std2' : [1,1], 'corr2' : 0.00,
#         }

df_data = make_dataset(params = scenario_di) 

df_resu = fit_rf_get_metrics(df_data, feat_li, rfo_n_trees = rfo_n_trees, random_seed = random_seed, max_features = max_features, max_depth = 30)

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



    