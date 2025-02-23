#--------------------             
# Author : Serge Zaugg
# Description : Create datasets, train, assess performance and feature importance
#--------------------

import numpy as np
import plotly.express as px
from utils import make_dataset, fit_rf_get_metrics
import plotly.graph_objects as go
from plotly.subplots import make_subplots

rfo_n_trees = 30
random_seed = 557
np.random.seed(seed=random_seed)

#-----------------------------
# (1) Define several scenarios 

scenarios_di = { 
    "Both features are informative" : {
        'n1' : 10000, 'mu1' : [0.0, 2.0] , 'std1' : [1,1], 'corr1' : 0.00,
        'n2' : 10000, 'mu2' : [2.0, 0.0] , 'std2' : [1,1], 'corr2' : 0.00,
        }
    ,
    "Both features are informative and redundant" : {
        'n1' : 10000, 'mu1' : [ 1.4,  1.4] , 'std1' : [1,1], 'corr1' : +0.98,
        'n2' : 10000, 'mu2' : [-1.4, -1.4] , 'std2' : [1,1], 'corr2' : +0.98,
        }
    ,
    "Joint information from features needed - parallel" : {
        'n1' : 10000, 'mu1' : [-0.14, -0.14] , 'std1' : [1,1], 'corr1' : -0.98,
        'n2' : 10000, 'mu2' : [+0.14, +0.14] , 'std2' : [1,1], 'corr2' : -0.98,
        }
    ,
    "Features are NOT informative" : {
        'n1' : 10000, 'mu1' : [0.0, 0.0] , 'std1' : [1,1], 'corr1' : -0.90,
        'n2' : 10000, 'mu2' : [0.0, 0.0] , 'std2' : [1,1], 'corr2' : -0.90,
        }
    , 
    "Only one features is informative " : {
        'n1' : 10000, 'mu1' : [ 1.0, 1.0] , 'std1' : [1,1], 'corr1' : 0.00,
        'n2' : 10000, 'mu2' : [-1.0, 1.0] , 'std2' : [1,1], 'corr2' : 0.00,
        }
    ,
    "Joint information from features needed - cross" : {
        'n1' : 10000, 'mu1' : [0.0, 0.0] , 'std1' : [1,1], 'corr1' : -0.98,
        'n2' : 10000, 'mu2' : [0.0, 0.0] , 'std2' : [1,1], 'corr2' : +0.98,
        }
    ,
    }


#--------------------------------
# (2) define several feature sets
feat_li = [
    ["f01", "f02", "f03"],
    ["f01", "f03"],
    ["f02", "f03"],
    ["f01", "f02"],
    ["f03"],
    ]


#-----------------------------------------
# (3) loop over scenarios and feature sets
for k in scenarios_di:
    print(k)
    mvn_params = scenarios_di[k]
    df = make_dataset(params = mvn_params) 
    df_resu = fit_rf_get_metrics(df, feat_li, rfo_n_trees = rfo_n_trees, random_seed = random_seed, max_features = 1, max_depth = 30)

    fig1 = px.scatter(
        data_frame = df,
        x = 'f01',
        y = 'f02',
        color = 'class',
        width = 500,
        height = 500,
        title = k,
        color_discrete_sequence=['#ee33ff', '#33aaff'],
        )
    _ = fig1.update_xaxes(showline = True, linecolor = 'white', linewidth = 1, row = 1, col = 1, mirror = True)
    _ = fig1.update_yaxes(showline = True, linecolor = 'white', linewidth = 1, row = 1, col = 1, mirror = True)
    fig1.update_traces(marker=dict(size=5))
    # fig1.show()

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
    fig.show()
