from fastapi import Request

from src.common.logging import logger


async def logging_middleware(request: Request, call_next) -> object:
    """Log request method + path on entry, status code on exit."""
    logger.info("Request", method=request.method, path=request.url.path)
    response = await call_next(request)
    logger.info("Response", status_code=response.status_code)
    return response
