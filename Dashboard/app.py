import streamlit as st
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)


st.set_page_config(
    page_title="Dashboard",
    layout="wide"
)

st.title("Analytics Governance Dashboard")
st.caption("Impact analysis and root-cause visibility for analytics changes")
