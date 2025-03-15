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

        A ranking of 'feature importance' is often desired.
        Here, I provide some interactive simulations and visuals to explain the limitations of 'feature importance'. 
        Synthetic datasets for supervised classification are created with one binary class (the target) and 3 continuous features (the predictors).
        The first two features (f01 and f02) can be manually tuned to be informative for classification, while the third (f03) is always non-informative.
        Random Forest classifiers are trained with all 3 features and also for smaller subsets of the features (f01 or f02 kicked-out).
        The predictive performance (ROC-AUC) is obtained from a test set and the **impurity-based feature importance** is computed.
        ''')

    with st.container(border=True, key='conta_03'):
        st.markdown(''':blue[**RESULTS**]''')
        st.markdown(
        '''    
        The first 4 predefined scenarios gave similar feature importance of the full model, approx. (0.45, 0.45, 0.10).
        However, the impact of removing f01 or f02 is very different!
        It can be concluded that individual feature importance does not give the full picture.
        It is better to focus on feature sets and how they affect predictive performance.
        ''')
        st.page_link("st_page_02.py", label="To perform experiments see **:blue[Interactive]**")


    with st.container(border=True, key='conta_04'):
        st.markdown(''':blue[**DETAILS**]''')
        st.markdown(
        '''    
        * N is the total sample size which is then split by half into train and test set
        * Based on sklearn
        ''')    

    
    
  

   