#--------------------             
# Author : Serge Zaugg
# Description : General info for the Streamlit frontend
#--------------------

import streamlit as st

col_aa, col_bb, = st.columns([0.60, 0.40])

with col_aa: 
    
    with st.container(border=True, key='conta_01'):
        st.title(':blue[Can we rank features by importance?]')
        st.subheader(":blue[Applied Machine Learning  ---  ML Tutorials  ---  Supervised Classification]") 
        # st.page_link("st_page_02.py", label="LINK : Interactive dashboard")
    
    with st.container(border=True, key='conta_02'):
        st.markdown(
        '''    
        :blue[**SUMMARY**]

        People often wish a ranking of feature importance. 
        So here I provide some visuals to help explain this. 
        Many scenarios can be assessed by manually defining each class distribution with sliders.
        Synthetic datasets for supervised classification are created with one binary class (the target) and 3 continuous features (the predictors).
        The first two features (f01 and f02) can be informative for classification, while the third (f03) is always non-informative.
        How the first two features inform classification can be actively chosen.
        Random Forest classifiers are trained for 'all 3 features' and for smaller subsets of the features.
        The predictive performance (ROC-AUC) is obtained from a test set and the **impurity-based feature importance** is computed.
        ''')

    with st.container(border=True, key='conta_03'):
        st.markdown(''':blue[**RESULTS**]''')

        st.page_link("st_page_01.py", label="For main results see **:blue[Predefined scenarios]**")
        st.page_link("st_page_02.py", label="To perform new experiments see **:blue[Define new scenarios]**")
        st.markdown(
        '''    
        The 4 predefined scenarios gave similar feature importance of the full model, approx. (0.45, 0.45, 0.10).
        However, the impact of removing f01 or f02 is very different!
        Thinking solely in terms of individual feature importance is not sufficient.
        It is better to focus on feature sets and how they affect predictive performance (e.g. Test AUC)
        ''')

    
    
  

   