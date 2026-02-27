import uvicorn
from fastapi import Depends, FastAPI, File, UploadFile
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.middleware import logging_middleware, verify_api_key
from src.common.logging import logger, setup_logging
from src.inference_pipeline.infer import InferencePipeline

setup_logging()

app = FastAPI(
    title="Brain Tumor MRI Classifier API",
    description="Production inference service for MRI classification."
)

app.add_middleware(BaseHTTPMiddleware, dispatch=logging_middleware)

inference_engine = None


@app.on_event("startup")
async def startup_initialization():
    """
    Warms up the application by initializing the deep learning engine.
    """
    global inference_engine
    logger.info("FastAPI service lifecycle: STARTUP")
    inference_engine = InferencePipeline()


@app.get("/health")
async def get_health_status():
    """
    Standard health check endpoint for monitoring tools and orchestrators.

    Returns:
        dict: Status message indicating the service is operational.
    """
    return {"status": "healthy"}


@app.post("/predict")
async def execute_prediction(
    uploaded_image: UploadFile = File(...),
    _auth: str = Depends(verify_api_key)
):
    """
    Processes an uploaded MRI scan and returns a binary classification.

    Args:
        uploaded_image (UploadFile): The JPEG or PNG image to classify.
        _auth (str): Authenticated API key status.

    Returns:
        dict: The classification label and tumor probability.
    """
    try:
        binary_payload = await uploaded_image.read()
        prediction_result = inference_engine.predict(binary_payload)

        logger.info(
            "Classification successful",
            label=prediction_result['label']
        )

        return prediction_result

    except Exception as api_error:
        logger.error("Inference endpoint failed", error=str(api_error))
        return {"error": "Internal processing error during inference."}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
