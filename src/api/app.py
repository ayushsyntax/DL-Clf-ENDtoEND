import uvicorn
import tensorflow as tf
from fastapi import Depends, FastAPI, File, UploadFile
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.middleware import logging_middleware, verify_api_key
from src.common.logging import logger, setup_logging
from src.common.config import settings

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
    model_path = settings.ARTIFACTS_DIR / "best_model.keras"
    if model_path.exists():
        try:
            inference_engine = tf.keras.models.load_model(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error("Failed to load model", error=str(e))
    else:
        logger.error("Model not found at startup", path=str(model_path))


@app.get("/health")
async def get_health_status():
    """
    Standard health check endpoint for monitoring tools and orchestrators.

    Returns:
        dict: Status message indicating the service is operational.
    """
    if inference_engine is not None:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "degraded", "model_loaded": False}


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
    if inference_engine is None:
        return {"error": "Model is not loaded."}

    try:
        binary_payload = await uploaded_image.read()

        # Preprocess exactly as in training
        decoded_image = tf.image.decode_image(binary_payload, channels=settings.CHANNELS, expand_animations=False)
        resized_image = tf.image.resize(
            decoded_image,
            [settings.IMAGE_SIZE, settings.IMAGE_SIZE]
        )
        preprocessed_image = tf.keras.applications.efficientnet_v2.preprocess_input(
            resized_image
        )

        # Expand dims for batch size 1
        input_tensor = tf.expand_dims(preprocessed_image, 0)

        # Predict
        prediction = inference_engine.predict(input_tensor, verbose=0)
        probability = float(prediction[0][0])

        # Read threshold or fallback
        threshold = getattr(settings, 'INFERENCE_THRESHOLD', 0.1723)

        if probability >= threshold:
            label = "Tumor"
            confidence = probability
        else:
            label = "No Tumor"
            confidence = 1.0 - probability

        logger.info(
            "Classification successful",
            label=label,
            probability=probability,
            confidence=confidence
        )

        return {
            "prediction": label,
            "probability": probability,
            "confidence": confidence
        }

    except (tf.errors.InvalidArgumentError, ValueError) as format_error:
        logger.error("Invalid image format or corrupt upload", error=str(format_error))
        return {"error": "Invalid or corrupt image payload."}
    except Exception as api_error:
        logger.error("Inference endpoint failed", error=str(api_error))
        return {"error": "Internal processing error during inference."}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
