#--------------------             
# Author : Serge Zaugg
# Description : Create datasets, train, assess performance and feature importance
#--------------------

import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from utils import make_dataset, fit_rf_get_metrics
from sklearn.utils import shuffle
from streamlit import session_state as ss

feat_li = [
    ["f01", "f02", "f03"],
    ["f01", "f03"],
    ["f02", "f03"],
    ]

#--------------------------------
# streamlit start here 

# initialize session state 
if 'rfo_n_trees' not in ss:
    ss['rfo_n_trees'] = 30
if 'max_features' not in ss:
    ss['max_features'] = 1
if 'max_depth' not in ss:
    ss['max_depth'] = 30
if 'random_seed' not in ss:
    ss['random_seed'] = 503

np.random.seed(seed=ss['random_seed'])


#----------------
# 1st line 
col_a, col_space01, col_c, col_d, = st.columns([0.20, 0.01, 0.50, 0.5])

preset_options = [
    "Both feat inform (hi-corr)",
    "Both feat inform (lo-corr)",
    "Jointly inform (parallel)",
    "Jointly inform (cross)",
    ]

with col_a:
    st.subheader("Select")
    option1 = st.selectbox("", preset_options, key = 'sel02')

if option1 == preset_options[0]:
    scenario_di = {
        'n1' : 10000, 'mu1' : [ 1.4,  1.4] , 'std1' : [1.0,1.0], 'corr1' : +0.98,
        'n2' : 10000, 'mu2' : [-1.4, -1.4] , 'std2' : [1.0,1.0], 'corr2' : +0.98,
        }
if option1 == preset_options[1]:
    scenario_di = {
        'n1' : 10000, 'mu1' : [0.0, 2.0] , 'std1' : [1.0,1.0], 'corr1' : 0.00,
        'n2' : 10000, 'mu2' : [2.0, 0.0] , 'std2' : [1.0,1.0], 'corr2' : 0.00,
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

df_data = make_dataset(params = scenario_di) 

df_data = shuffle(df_data)

df_resu = fit_rf_get_metrics(df_data, feat_li, rfo_n_trees = ss['rfo_n_trees'], random_seed = ss['random_seed'], max_features = ss['max_features'], max_depth = ss['max_depth'])

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
    st.subheader("Predictive performance vs importance")
    # reshape dfs
    df_r_num = df_resu.iloc[:,0:4].astype(float)
    df_r_num['Included_Features'] = df_resu['Included_Features']
    df_long = pd.melt(df_r_num, id_vars='Included_Features', value_vars=['AUC_Test', 'Importance_f01', 'Importance_f02', 'Importance_f03'])
    df_imp = df_long[df_long["variable"] != 'AUC_Test']
    df_auc = df_long[df_long["variable"] == 'AUC_Test']
    # print(df_resu)
    # print(df_r_num)
    # print(df_long)
    df_imp.fillna(value=0.0, inplace=True)
    df_auc.fillna(value=0.0, inplace=True)
    # AUC figure
    barfig1 = px.bar(
        df_auc, 
        x="Included_Features", 
        y="value", 
        color="variable", 
        text_auto=True, 
        barmode='group', 
        width = 600, 
        height = 290,
        labels={"value": "ROC-AUC", }, 
        color_discrete_sequence = ss['bar_colors_1']
        )
    barfig1.update_layout(yaxis_range=[0.0,1.0])
    _ = barfig1.update_xaxes(showline = True, linecolor = 'white', linewidth = 2, row = 1, col = 1, mirror = True)
    _ = barfig1.update_yaxes(showline = True, linecolor = 'white', linewidth = 2, row = 1, col = 1, mirror = True)
    _ = barfig1.update_layout(paper_bgcolor="#112233",)
    _ = barfig1.update_layout(margin=dict(r=180, t=40 ))
    _ = barfig1.update_layout(legend=dict(yanchor="top", y=0.9, xanchor="left", x=1.1)) 
    # importance figure     
    barfig2 = px.bar(
        df_imp,
        x="Included_Features", 
        y="value", 
        color="variable", 
        text_auto=True, 
        barmode='group', 
        width = 600, 
        height = 290,
        labels={"value": "Feature importance", },
        color_discrete_sequence = ss['bar_colors_2']
        )
    barfig2.update_layout(yaxis_range=[0.0,1.0])
    _ = barfig2.update_xaxes(showline = True, linecolor = 'white', linewidth = 2, row = 1, col = 1, mirror = True)
    _ = barfig2.update_yaxes(showline = True, linecolor = 'white', linewidth = 2, row = 1, col = 1, mirror = True)
    _ = barfig2.update_layout(paper_bgcolor="#112233",)
    _ = barfig2.update_layout(margin=dict(r=180, t=40  ))
    _ = barfig2.update_layout(legend=dict(yanchor="top", y=0.9, xanchor="left", x=1.1)) 
    # show
    st.plotly_chart(barfig1, use_container_width=False)
    st.plotly_chart(barfig2, use_container_width=False)
    # st.dataframe(df_resu, hide_index=True) 


#----------------
# 2nd line 
st.divider() 
col01, col02, col03, col04, col05, col06= st.columns([0.10, 0.10, 0.10, 0.10, 0.50, 0.10,]) 
with col01:
    ss['rfo_n_trees']  = st.number_input("Nb trees", min_value=1, max_value=100,       value=30,  step=10)
with col02:
    ss['max_features'] = st.number_input("Max features", min_value=1, max_value=3,      value=1,   step=1)
with col03:
    ss['max_depth']    = st.number_input("Max tree depth", min_value=1,  max_value=50,  value=30,  step=1)
with col04:
    ss['random_seed']  = st.number_input("Random seed", min_value=1,  max_value=1000,   value=503, step=1)




    