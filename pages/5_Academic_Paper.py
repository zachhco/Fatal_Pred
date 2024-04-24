import streamlit as st
import base64

st.markdown("# Harnessing Predictive Models to Enhance Road Safety: ")
st.markdown("### An Econometric Investigation into Risk Compensation & A Test of Machine Learning Model Efficacy")\

# Path to your local PDF file
pdf_path = 'paper.pdf'

# Open the PDF file in binary read mode
with open(pdf_path, "rb") as pdf_file:
    base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')

pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="1000" type="application/pdf"></iframe>'

st.markdown(pdf_display, unsafe_allow_html=True)

