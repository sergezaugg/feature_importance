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

# np.random.seed(seed=ss['random_seed'])

feat_li = [
    ["f01", "f02", "f03"],
    ["f01", "f03"],
    ["f02", "f03"],
    ]



# Define pre-specified scenarios 
N = 3000
scenarios_presp = { 
    "Linearly separable I" : {
        'n1' : N, 'mu1' : [0.0, 2.0] , 'std1' : [1.1,1.1], 'corr1' : 0.00,
        'n2' : N, 'mu2' : [2.0, 0.0] , 'std2' : [1.0,1.0], 'corr2' : 0.00,
        },
    "Saurona" : {           
        'n1' : N, 'mu1' : [0.0, 0.0] , 'std1' : [1.2,1.2], 'corr1' : 0.0,
        'n2' : N, 'mu2' : [0.0, 0.0] , 'std2' : [0.05,0.7], 'corr2' : 0.0,
        },
    "Parallel" : {
        'n1' : N, 'mu1' : [-0.14, -0.14] , 'std1' : [1.2,1.2], 'corr1' : -0.98,
        'n2' : N, 'mu2' : [+0.14, +0.14] , 'std2' : [1.2,1.2], 'corr2' : -0.98,
        },
    "Cross" : {
        'n1' : N, 'mu1' : [0.0, 0.0] , 'std1' : [1.0, 1.0], 'corr1' : -0.96,
        'n2' : N, 'mu2' : [0.0, 0.0] , 'std2' : [1.0, 1.0], 'corr2' : +0.96,
        },
    "Linearly separable II" : {
        'n1' : N, 'mu1' : [ 1.0, 1.0] , 'std1' : [1.0,1.0], 'corr1' : 0.00,
        'n2' : N, 'mu2' : [-1.0, 1.0] , 'std2' : [1.0,1.0], 'corr2' : 0.00,
        },
    "Weak informative" : {
        'n1' : N, 'mu1' : [0.5, 0.0] , 'std1' : [1.0,1.0], 'corr1' : -0.90,
        'n2' : N, 'mu2' : [0.0, 0.0] , 'std2' : [1.0,1.0], 'corr2' : -0.90,
        }, 
   "Redundant" : {
        'n1' : N, 'mu1' : [ 1.4,  1.4] , 'std1' : [1.0,1.0], 'corr1' : +0.98,
        'n2' : N, 'mu2' : [-1.4, -1.4] , 'std2' : [1.0,1.0], 'corr2' : +0.98,
        }, 
   "Not separable" : {
        'n1' : N, 'mu1' : [0.0, 0.0] , 'std1' : [1.1,1.1], 'corr1' : 0.00,
        'n2' : N, 'mu2' : [0.0, 0.0] , 'std2' : [1.1,1.1], 'corr2' : 0.00,
        }, 
   "Looking up" : {
        'n1' : N, 'mu1' : [0.0, 0.0] , 'std1' : [1.0,1.0], 'corr1' : 0.0,
        'n2' : N, 'mu2' : [0.0, 1.0] , 'std2' : [0.15,0.1], 'corr2' : 0.0,
        }
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
    ss['distr'] = {'cus' : scenarios_presp['Linearly separable I']}     
    # ss['distr']['cus'] = scenarios_presp['Linearly separable I']       




# # reset 
# with st.form("reset_01", border=False):
#     submitted = st.form_submit_button("Reset")
#     if submitted: 
#         ss['distr'] = init_vals



with st.container(border=True, key='conta_b01'):
    with st.form(key = "f01", border=False):
        a0, a1, a2 = st.columns(3)  # st.columns([0.60, 0.40, 0.40])
        with a0:
            preset_options = ["Linearly separable I", 
                                "Linearly separable II", 
                                "Weak informative",
                                "Redundant",
                                "Parallel", 
                                "Cross", 
                                "Saurona", 
                                "Looking up",
                                "Not separable",
                                ]
            option1 = st.selectbox("Predefined distributions", preset_options, key = 'sel02')
        with a1:
            st.text("")
            st.text("")
            submitted = st.form_submit_button("Confirm")
        if submitted: 
            ss['distr'] = {'cus' : scenarios_presp[option1]}






#----------------
# 1st line 
col_a, col_b, col_space01, col_c, col_d, = st.columns([0.10, 0.10, 0.05, 0.50, 0.5])

with col_a:
    st.subheader("Class A")
    # with st.form("rand_1", border=False):
    #     submitted = st.form_submit_button("Randomize")
    #     if submitted: 
    #         ss['distr']['mean1x'] = np.random.uniform(low=-5.0,  high=+5.0, size=1)[0]
    #         ss['distr']['mean1y'] = np.random.uniform(low=-5.0,  high=+5.0, size=1)[0]
    #         ss['distr']['stdv1x'] = np.random.uniform(low= 0.01, high=5.0, size=1)[0]
    #         ss['distr']['stdv1y'] = np.random.uniform(low=-0.01, high=5.0, size=1)[0]
    #         ss['distr']['corr1']  = np.random.uniform(low=-1.0,  high=+1.0, size=1)[0]
    numb1  = st.slider("N",       min_value=  10,   max_value=1000, value=300,  key="slide_n1")
    mean1x = st.slider("mean x",  min_value= -5.0,  max_value=+5.0, value=ss['distr']['cus']['mu1'][0], key="slide_mu1x")
    mean1y = st.slider("mean y",  min_value= -5.0,  max_value=+5.0, value=ss['distr']['cus']['mu1'][1], key="slide_mu1y")
    stdv1x = st.slider("stdev x", min_value= +0.01, max_value=10.0, value=ss['distr']['cus']['std1'][0], key="slide_std1x")
    stdv1y = st.slider("stdev y", min_value= +0.01, max_value=10.0, value=ss['distr']['cus']['std1'][1], key="slide_std1y")
    corr1  = st.slider("corr",    min_value=-1.0,   max_value=+1.0, value=ss['distr']['cus']['corr1']  , key="slide_corr1")
with col_b:   
    st.subheader("Class B")
    # with st.form("rand_2", border=False):
    #     submitted = st.form_submit_button("Randomize")
    #     if submitted: 
    #         ss['distr']['mean2x'] =  np.random.uniform(low=-5.0,  high=+5.0, size=1)[0]
    #         ss['distr']['mean2y'] =  np.random.uniform(low=-5.0,  high=+5.0, size=1)[0]
    #         ss['distr']['stdv2x'] =  np.random.uniform(low= 0.01, high=5.0, size=1)[0]
    #         ss['distr']['stdv2y'] =  np.random.uniform(low=-0.01, high=5.0, size=1)[0]
    #         ss['distr']['corr2']  =  np.random.uniform(low=-1.0,  high=+1.0, size=1)[0]
    numb2  = st.slider("N",       min_value=  10,   max_value=1000, value=300,  key="slide_n2")
    mean2x = st.slider("mean x",  min_value= -5.0,  max_value=+5.0, value=ss['distr']['cus']['mu2'][0], key="slide_mu2x")
    mean2y = st.slider("mean y",  min_value= -5.0,  max_value=+5.0, value=ss['distr']['cus']['mu2'][1], key="slide_mu2y")
    stdv2x = st.slider("stdev x", min_value= +0.01, max_value=10.0, value=ss['distr']['cus']['std2'][0], key="slide_std2x")
    stdv2y = st.slider("stdev y", min_value= +0.01, max_value=10.0, value=ss['distr']['cus']['std2'][1], key="slide_std2y")
    corr2  = st.slider("corr",    min_value=-1.0,   max_value=+1.0, value=ss['distr']['cus']['corr2'] , key="slide_corr2")


#----------------
# computation block 

ss['distr']['cus'] = { 
# scenario_di = { 
        'n1' : numb1, 'mu1' : [mean1x, mean1y] , 'std1' : [stdv1x, stdv1y], 'corr1' : corr1,
        'n2' : numb2, 'mu2' : [mean2x, mean2y] , 'std2' : [stdv2x, stdv2y], 'corr2' : corr2,
        }

#  'n1' : N, 'mu1' : [0.0, 2.0] , 'std1' : [1.1,1.1], 'corr1' : 0.00,
#         'n2' : N, 'mu2' : [2.0, 0.0] , 'std2' : [1.0,1.0], 'corr2' : 0.00,

df_data = make_dataset(params = ss['distr']['cus']) 
df_data = shuffle(df_data)
df_resu = fit_rf_get_metrics(df_data, feat_li, rfo_n_trees = ss['rfo_n_trees'], random_seed = ss['random_seed'], max_features = ss['max_features'], max_depth = ss['max_depth'])
# to enforce same class order in plots 
df_data = df_data.sort_values(by='class')

fig1 = px.scatter(
    data_frame = df_data,
    x = 'f01',
    y = 'f02',
    color = 'class',
    width = 600,
    height = 600,
    title = "",
    color_discrete_sequence=['#22ff99', '#9911ff']
    )
_ = fig1.update_xaxes(showline = True, linecolor = 'white', linewidth = 1, row = 1, col = 1, mirror = True)
_ = fig1.update_yaxes(showline = True, linecolor = 'white', linewidth = 1, row = 1, col = 1, mirror = True)
_ = fig1.update_layout(paper_bgcolor="#112233",)
_ = fig1.update(layout_xaxis_range = [-15,+15])
_ = fig1.update(layout_yaxis_range = [-15,+15])
fig1.update_traces(marker=dict(size=5))
#----------------


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
    barfig1 = px.bar(df_auc, x="Included_Features", y="value", color="variable", text_auto=True, barmode='group', 
                     width = 600, height = 290,
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
    barfig2 = px.bar(df_imp, 
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
    _ = barfig2.update_layout(margin=dict(r=180, t=40 ))
    _ = barfig2.update_layout(legend=dict(yanchor="top", y=0.9, xanchor="left", x=1.1)) 
    # show on dashboard
    st.plotly_chart(barfig1, use_container_width=False)
    st.plotly_chart(barfig2, use_container_width=False)


#----------------
# 2nd line 
st.divider() 
col01, col02, col03, col04, col05, col06= st.columns([0.10, 0.10, 0.10, 0.10, 0.50, 0.10,]) 
with col01:
    ss['rfo_n_trees']  = st.number_input("Nb trees", min_value=10, max_value=100,       value=30,  step=10)
with col02:
    ss['max_features'] = st.number_input("Max features", min_value=1, max_value=3,      value=1,   step=1)
with col03:
    ss['max_depth']    = st.number_input("Max tree depth", min_value=1,  max_value=50,  value=30,  step=1)
with col04:
    ss['random_seed']  = st.number_input("Random seed", min_value=1,  max_value=1000,   value=503, step=1)



