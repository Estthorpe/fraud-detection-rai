import streamlit as st

st.set_page_config(page_title="Fraud Detection + RAI", layout="wide")

st.title("Fraud Detection with Responsible AI Checks")
tab1, tab2, tab3, tab4 = st.tabs(["Scoring", "Metrics", "Explainability", "Fairness"])

with tab1:
    st.info("CSV upload & threshold slider will appear here (Phase 7).")

with tab2:
    st.info("Model metrics, ROC/PR curves summary (Phase 2-4).")

with tab3:
    st.info("Global SHAP + local case explanations (Phase 5).")

with tab4:
    st.info("Synthetic group fairness metrics & parity gaps (Phase 6).")
