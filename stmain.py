#--------------------             
# Author : Serge Zaugg
# Description : Main streamlit entry point
#--------------------

import streamlit as st

st.set_page_config(layout="wide")

p0 = st.Page("st_page_00.py", title="Summary")
p1 = st.Page("st_page_02.py", title="Define new scenarios")
p2 = st.Page("st_page_01.py", title="Predefined scenarios")

pg = st.navigation([p0, p1, p2])

pg.run()

# run locally
# streamlit run stmain.py


