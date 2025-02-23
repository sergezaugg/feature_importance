#--------------------             
# Author : Serge Zaugg
# Description : Main streamlit entry point
#--------------------
import streamlit as st

st.set_page_config(layout="wide")

p0 = st.Page("stmain0.py", title="Summary")
p1 = st.Page("stmain2.py", title="Define new scenarios")
p2 = st.Page("stmain1.py", title="Predefined scenarios")

pg = st.navigation([p0, p1, p2])

pg.run()

# streamlit run stmain.py


