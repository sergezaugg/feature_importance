#--------------------             
# Author : Serge Zaugg
# Description : General info for the Streamlit frontend
#--------------------

import streamlit as st

col_aa, col_bb, = st.columns([0.60, 0.40])

with col_aa: 
    st.title('Can we rank features by importance?')
    st.markdown(
    '''    
    :violet[**SUMMARY:**]

    This dashboard is primarily didactic. 
    People often wish a ranking of feature importance. 
    So here I provide some visuals to explain this. 
    Many scenarios can be assessed by manually defining each class distribution with sliders.

    :violet[**METHODS:**]

    Synthetic datasets for supervised classification are created with one binary class (the target) and 3 continuous features (the predictors).
    The first two features (f01 and f02) can be informative for classification, while the third (f03) is always non-informative.
    How the first two features inform classification can be actively chosen.
    Random Forest classifiers are trained for 'all 3 features' and for smaller subsets of the features.
    The predictive performance (ROC-AUC) is obtained from a test set and the **impurity-based feature importance** is computed.
    ''')

    st.markdown(''':violet[**RESULTS:**]''')

    st.page_link("st_page_01.py", label="For main results see **Predefined scenarios**")

    st.page_link("st_page_02.py", label="To perform new experiments see **Define new scenarios**")
    
    st.markdown(
    '''    
    :violet[**DISCUSSION:**]

    The 4 predefined scenarios gave similar feature importance of the full model, approx. (0.45, 0.45, 0.10).
    However, the impact of removing f01 or f02 is very different!
    Thinking solely in terms of individual feature importance is not sufficient.
    It is better to focus on feature sets and how they affect predictive performance (e.g. Test AUC)
    ''')

   