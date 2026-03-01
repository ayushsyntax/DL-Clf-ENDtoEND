import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.middleware import logging_middleware
from src.common.logging import logger, setup_logging
from src.inference_pipeline.infer import InferencePipeline

setup_logging()

app = FastAPI(
    title="Brain Tumor MRI Classifier API",
    description="Binary inference: Tumor vs No Tumor from MRI scans.",
    version="1.0.0"
)

app.add_middleware(BaseHTTPMiddleware, dispatch=logging_middleware)

inference_engine: InferencePipeline = None


@app.on_event("startup")
async def startup_initialization() -> None:
    """Load model once at startup; S3 fallback handles ECS cold starts."""
    global inference_engine
    try:
        inference_engine = InferencePipeline()
        logger.info("Inference engine ready")
    except Exception as e:
        logger.error("Model load failed at startup", error=str(e))


@app.get("/health")
async def health_check() -> dict:
    """Liveness probe for ECS, ALB, and uptime monitors."""
    if inference_engine is not None:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "degraded", "model_loaded": False}


@app.post("/predict")
async def predict(uploaded_image: UploadFile = File(...)) -> dict:
    """
    Classify uploaded MRI image as Tumor or No Tumor.

    Returns:
        label (str), probability (float), class_idx (0 or 1)
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = inference_engine.predict(await uploaded_image.read())
        logger.info("Prediction complete", **result)
        return result
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid or corrupt image.")
    except Exception as e:
        logger.error("Inference failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal inference error.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
