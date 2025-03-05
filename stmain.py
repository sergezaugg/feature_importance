#--------------------             
# Author : Serge Zaugg
# Description : Main streamlit entry point
# run locally : streamlit run stmain.py
#--------------------

import streamlit as st
from streamlit import session_state as ss

# stuff shared across pages 
if 'bar_colors_1' not in ss:
    ss['bar_colors_1'] = ['#0000ff']
if 'bar_colors_2' not in ss:
    ss['bar_colors_2'] = ['#0055ff', '#0077dd', '#0099bb']


st.set_page_config(layout="wide")

p0 = st.Page("st_page_00.py", title="Summary")
p1 = st.Page("st_page_02.py", title="Define new scenarios")
p2 = st.Page("st_page_01.py", title="Predefined scenarios")

pg = st.navigation([p0, p1, p2])

pg.run()


with st.sidebar:
    st.text("-- Under development --")
    st.title(""); st.title(""); st.title(""); st.title(""); st.title(""); st.title(""); st.title("")
    st.title(""); st.title(""); st.title(""); st.title("") 
    st.markdown(''':gray[RELATED TOPICS]''')
    st.page_link("https://ml-performance-metrics.streamlit.app/", label=":gray[ml-performance-metrics]")
    st.page_link("https://purenoisefeatures.streamlit.app//", label=":gray[pure-noise-features]")

