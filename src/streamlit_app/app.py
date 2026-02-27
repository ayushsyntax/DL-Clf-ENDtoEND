import requests
import streamlit as st

from src.common.config import settings


def run_streamlit_demo():
    """
    Constructs and executes the Streamlit UI for local MRI analysis.

    This interface is strictly designed for text-based predictions.
    Image display and visualizations are excluded per production spec.
    """
    st.set_page_config(page_title="Brain Tumor MRI Detector", page_icon="🧠")

    st.title("🧠 Brain Tumor MRI Classifier")
    st.write("Upload an MRI scan (.jpg, .jpeg, .png) for analysis.")

    uploaded_file = st.file_uploader(
        "Select MRI Scan",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        if st.button("Predict"):
            execute_prediction_workflow(uploaded_file)

    st.info("Note: Production demo - Prediction result and confidence only.")


def execute_prediction_workflow(file_buffer):
    """
    Coordinates the API communication and result display.

    Args:
        file_buffer (st.runtime.uploaded_file_manager.UploadedFile): Loaded image.
    """
    with st.spinner('Analyzing payload...'):
        try:
            api_endpoint = "http://localhost:8000/predict"
            files_payload = {
                "uploaded_image": (
                    file_buffer.name,
                    file_buffer.getvalue(),
                    file_buffer.type
                )
            }
            auth_headers = {"X-API-KEY": settings.API_KEY}

            http_response = requests.post(
                api_endpoint,
                files=files_payload,
                headers=auth_headers
            )

            if http_response.status_code == 200:
                prediction_data = http_response.json()
                label = prediction_data['label']
                probability = prediction_data['probability']

                st.subheader(f"Result: {label}")
                st.metric("Confidence Score", f"{probability:.2%}")
            else:
                st.error(f"API Error: {http_response.status_code}")

        except Exception as network_error:
            st.error(f"Backend unreachable: {str(network_error)}")


if __name__ == "__main__":
    run_streamlit_demo()
