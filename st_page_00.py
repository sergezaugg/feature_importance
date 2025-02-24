#--------------------             
# Author : Serge Zaugg
# Description : some info 
#--------------------

import streamlit as st

col_aa, col_bb, = st.columns([0.60, 0.40])

with col_aa: 
    st.title('Can we rank features by importance?')
    st.markdown(
    '''    
    :violet[**SUMMARY:**]

    :blue[This dashboard is primarily didactic. 
    People often wish a ranking of feature importance. 
    So here I provide some visuals to explain this. 
    Many scenarios can be assessed by manually defining each class distribution with sliders.
    Details see : https://github.com/sergezaugg/feature_importance.]

    :violet[**METHODS:**]

    :blue[Synthetic datasets for supervised classification are created with one binary class (the target) and 3 continuous features (the predictors).
    The first two features (f01 and f02) can be informative for classification, while the third (f03) is always non-informative.
    How the first two features inform classification can be actively chosen.
    Random Forest classifiers are trained for 'all 3 features' and for smaller subsets of the features.
    The predictive performance (ROC-AUC) is obtained from a test set and the **impurity-based feature importance** is computed.] 

    :violet[**RESULTS:**]

    :blue[For main results see *Predefined scenarios* ]

    :blue[To perform new experiments see *Define new scenarios*]

    ''')

    st.markdown(
    '''    
    :violet[**DISCUSSION:**]

    :blue[
    The 4 predefined scenarios gave similar feature importance of the full model, approx. (0.45, 0.45, 0.10).
    However, the impact of removing f01 or f02 is very different!
    Thinking solely in terms of individual feature importance does often not make sense.
    It is better to focus on feature sets and how they affect predictive performance (e.g. Test AUC)]
    ''')

   