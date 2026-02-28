import os
import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="centered")

col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title("Brain Tumor MRI Classifier")

# Health check
try:
    health_res = requests.get(f"{API_URL}/health", timeout=5)
    if health_res.status_code == 200 and health_res.json().get("status") == "healthy":
        with col2:
            st.success("🟢 API Online")
    else:
        with col2:
            st.warning("🟡 API Degraded")
except Exception:
    with col2:
        st.error("🔴 API Offline")

st.write(f"Connected to: `{API_URL}`")

uploaded_file = st.file_uploader("Upload an MRI image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded MRI Scan", use_column_width=True)
    if st.button("Predict"):
        with st.spinner("Analyzing scan..."):
            try:
                # Send as a file tuple to ensure proper multipart/form-data encoding
                files = {
                    "uploaded_image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }
                response = requests.post(f"{API_URL}/predict", files=files)
                if response.status_code == 200:
                    result = response.json()
                    label = result.get("prediction", "Unknown")
                    confidence = float(result.get("confidence", 0.0))

                    st.markdown(f"### Result: **{label}**")
                    st.progress(confidence)
                    st.write(f"Confidence: {confidence:.2%}")
                else:
                    st.error(f"Error from API: {response.text}")
            except Exception as e:
                st.error(f"Prediction request failed: {e}")
