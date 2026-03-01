import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.environ.get("API_URL", "http://localhost:8000")


def main() -> None:
    """Simple UI wrapper around FastAPI MRI classifier."""
    st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="centered")

    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.title("Brain Tumor MRI Classifier")

    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        status = health.json().get("status")
        with col2:
            if health.status_code == 200 and status == "healthy":
                st.success("🟢 API Online")
            else:
                st.warning("🟡 API Degraded")
    except Exception:
        with col2:
            st.error("🔴 API Offline")

    st.write(f"Connected to: `{API_URL}`")

    uploaded = st.file_uploader(
        "Upload an MRI image (JPG/PNG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded is None:
        return

    st.image(uploaded, caption="Uploaded MRI Scan", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing scan..."):
            try:
                files = {
                    "uploaded_image": (
                        uploaded.name,
                        uploaded.getvalue(),
                        uploaded.type or "image/jpeg",
                    )
                }
                resp = requests.post(f"{API_URL}/predict", files=files, timeout=30)
                if resp.status_code != 200:
                    st.error(f"Error from API: {resp.text}")
                    return

                payload = resp.json()
                label = payload.get("label", "Unknown")
                prob = float(payload.get("probability", 0.0))

                st.markdown(f"### Result: **{label}**")
                st.progress(prob)
                st.write(f"Model probability (tumor): {prob:.2%}")

            except Exception as exc:
                st.error(f"Prediction request failed: {exc}")


if __name__ == "__main__":
    main()
