"""Streamlit entry point: two pages, Documents and Evaluation."""
import streamlit as st

pages = [
    st.Page("views/documents.py", title="Documents", icon="📄", default=True),
    st.Page("views/evaluation.py", title="Evaluation", icon="📊"),
]
st.navigation(pages).run()
