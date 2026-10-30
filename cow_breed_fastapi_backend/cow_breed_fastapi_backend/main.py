
"""FastAPI backend for the cow/buffalo breed classifier."""
from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("breed-api")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "cow_breed_classifier.keras")))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))

# This ordering matches the class_names list in the supplied predict script and
# the alphabetical directory ordering used by Keras flow_from_directory.
CLASS_NAMES = [
    "Gir",
    "Holstein_Friesian",
    "Jaffrabadi",
    "Jersey",
    "Murrah",
    "Sahiwal",
]
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
model: Any = None
prediction_lock = Lock()


def load_classifier() -> Any:
    """Load and validate the trained TensorFlow/Keras model once at startup."""
    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}. Put the .keras model beside main.py "
            "or set MODEL_PATH to its full path."
        )

    loaded = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    output_shape = loaded.output_shape
    output_count = output_shape[-1] if isinstance(output_shape, (tuple, list)) else None
    if output_count is not None and int(output_count) != len(CLASS_NAMES):
        raise RuntimeError(
            f"Model has {output_count} output classes, but {len(CLASS_NAMES)} class labels are configured."
        )
    logger.info("Loaded model from %s", MODEL_PATH)
    return loaded


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_classifier()
    yield
    model = None


app = FastAPI(
    title="Cow & Buffalo Breed Classifier API",
    description=(
        "Upload a cow or buffalo image to predict one of the trained breeds: "
        "Gir, Holstein Friesian, Jaffrabadi, Jersey, Murrah, or Sahiwal. "
        "A breed is returned only when model confidence is strictly above 75%."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for a separately served local frontend (VS Code Live Server, Vite,
# or Python's http.server). Set CORS_ORIGINS to a comma-separated allowlist
# for deployment. Localhost ports are allowed by default for development.
DEFAULT_CORS_ORIGINS = ",".join(
    [
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]
)
origins_env = os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
allowed_origins = [origin.strip().rstrip("/") for origin in origins_env.split(",") if origin.strip()]
allow_localhost_ports = os.getenv("CORS_ALLOW_LOCALHOST_PORTS", "true").lower() in {
    "1", "true", "yes", "on"
}
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=(r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$" if allow_localhost_ports else None),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def root() -> dict[str, str]:
    return {
        "message": "Cow & Buffalo Breed Classifier API is running",
        "docs": "/docs",
        "health": "/health",
        "prediction_endpoint": "POST /predict",
    }


@app.get("/health", tags=["General"])
def health() -> dict[str, Any]:
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "model_file": MODEL_PATH.name,
        "classes": CLASS_NAMES,
    }


def _predict_image(image_bytes: bytes) -> dict[str, Any]:
    """Decode an image and return the top prediction plus class probabilities."""
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="The model is not loaded. Check server logs.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format not in ALLOWED_FORMATS:
                raise HTTPException(
                    status_code=415,
                    detail="Unsupported image format. Upload a JPG, PNG, or WEBP image.",
                )
            image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            pixels = np.asarray(image, dtype=np.float32) / 255.0
            batch = np.expand_dims(pixels, axis=0)
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc

    try:
        # Serialize model calls for predictable behaviour on small CPU deployments.
        with prediction_lock:
            raw_prediction = model.predict(batch, verbose=0)
        scores = np.asarray(raw_prediction)[0].astype(float)
        if scores.ndim != 1 or len(scores) != len(CLASS_NAMES):
            raise ValueError(f"Expected {len(CLASS_NAMES)} output scores; received shape {np.asarray(raw_prediction).shape}.")
        best_index = int(np.argmax(scores))
        best_confidence = float(scores[best_index])

        # Do not reveal a breed prediction when confidence is 75% or lower.
        # A value strictly greater than the threshold is required.
        if best_confidence <= CONFIDENCE_THRESHOLD:
            return {
                "success": False,
                "message": (
                    "Unable to identify the breed confidently. Please upload a clear, "
                    "correct image of a cow or buffalo and try again."
                ),
            }

        ranked_indices = np.argsort(scores)[::-1]
        return {
            "success": True,
            "predicted_breed": CLASS_NAMES[best_index],
            "confidence": round(best_confidence, 6),
            "confidence_percent": round(best_confidence * 100, 2),
            "top_predictions": [
                {
                    "breed": CLASS_NAMES[int(index)],
                    "confidence": round(float(scores[index]), 6),
                    "confidence_percent": round(float(scores[index]) * 100, 2),
                }
                for index in ranked_indices[:3]
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Model inference failed")
        raise HTTPException(status_code=500, detail="Prediction failed. Check server logs.") from exc


@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    """Predict the breed from one uploaded JPG, PNG, or WEBP image."""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file (JPG, PNG, or WEBP).")

    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()

    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image is too large. Maximum upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    result = await run_in_threadpool(_predict_image, content)
    result["filename"] = file.filename or "uploaded-image"
    return result




# -----------------------------------




# """FastAPI backend for the cow/buffalo breed classifier."""
# from __future__ import annotations

# import io
# import logging
# import os
# from contextlib import asynccontextmanager
# from pathlib import Path
# from threading import Lock
# from typing import Any

# import numpy as np
# import tensorflow as tf
# from fastapi import FastAPI, File, HTTPException, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from PIL import Image, UnidentifiedImageError
# from starlette.concurrency import run_in_threadpool

# logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
# logger = logging.getLogger("breed-api")

# BASE_DIR = Path(__file__).resolve().parent
# MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "cow_breed_classifier.keras")))
# IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
# MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
# CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))

# # This ordering matches the class_names list in the supplied predict script and
# # the alphabetical directory ordering used by Keras flow_from_directory.
# CLASS_NAMES = [
#     "Gir",
#     "Holstein_Friesian",
#     "Jaffrabadi",
#     "Jersey",
#     "Murrah",
#     "Sahiwal",
# ]
# ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
# model: Any = None
# prediction_lock = Lock()


# def load_classifier() -> Any:
#     """Load and validate the trained TensorFlow/Keras model once at startup."""
#     if not MODEL_PATH.is_file():
#         raise RuntimeError(
#             f"Model file not found: {MODEL_PATH}. Put the .keras model beside main.py "
#             "or set MODEL_PATH to its full path."
#         )

#     loaded = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
#     output_shape = loaded.output_shape
#     output_count = output_shape[-1] if isinstance(output_shape, (tuple, list)) else None
#     if output_count is not None and int(output_count) != len(CLASS_NAMES):
#         raise RuntimeError(
#             f"Model has {output_count} output classes, but {len(CLASS_NAMES)} class labels are configured."
#         )
#     logger.info("Loaded model from %s", MODEL_PATH)
#     return loaded


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global model
#     model = load_classifier()
#     yield
#     model = None


# app = FastAPI(
#     title="Cow & Buffalo Breed Classifier API",
#     description=(
#         "Upload a cow or buffalo image to predict one of the trained breeds: "
#         "Gir, Holstein Friesian, Jaffrabadi, Jersey, Murrah, or Sahiwal. "
#         "A breed is returned only when model confidence is strictly above 75%."
#     ),
#     version="1.0.0",
#     lifespan=lifespan,
# )

# # Convenient for local frontend development. Restrict origins in production.
# origins_env = os.getenv("CORS_ORIGINS", "*")
# allowed_origins = [origin.strip() for origin in origins_env.split(",") if origin.strip()]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=allowed_origins or ["*"],
#     allow_credentials=False,
#     allow_methods=["GET", "POST", "OPTIONS"],
#     allow_headers=["*"],
# )


# @app.get("/", tags=["General"])
# def root() -> dict[str, str]:
#     return {
#         "message": "Cow & Buffalo Breed Classifier API is running",
#         "docs": "/docs",
#         "health": "/health",
#         "prediction_endpoint": "POST /predict",
#     }


# @app.get("/health", tags=["General"])
# def health() -> dict[str, Any]:
#     return {
#         "status": "ok" if model is not None else "model_not_loaded",
#         "model_loaded": model is not None,
#         "model_file": MODEL_PATH.name,
#         "classes": CLASS_NAMES,
#     }


# def _predict_image(image_bytes: bytes) -> dict[str, Any]:
#     """Decode an image and return the top prediction plus class probabilities."""
#     global model
#     if model is None:
#         raise HTTPException(status_code=503, detail="The model is not loaded. Check server logs.")

#     try:
#         with Image.open(io.BytesIO(image_bytes)) as image:
#             image.verify()
#         with Image.open(io.BytesIO(image_bytes)) as image:
#             if image.format not in ALLOWED_FORMATS:
#                 raise HTTPException(
#                     status_code=415,
#                     detail="Unsupported image format. Upload a JPG, PNG, or WEBP image.",
#                 )
#             image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
#             pixels = np.asarray(image, dtype=np.float32) / 255.0
#             batch = np.expand_dims(pixels, axis=0)
#     except HTTPException:
#         raise
#     except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError) as exc:
#         raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc

#     try:
#         # Serialize model calls for predictable behaviour on small CPU deployments.
#         with prediction_lock:
#             raw_prediction = model.predict(batch, verbose=0)
#         scores = np.asarray(raw_prediction)[0].astype(float)
#         if scores.ndim != 1 or len(scores) != len(CLASS_NAMES):
#             raise ValueError(f"Expected {len(CLASS_NAMES)} output scores; received shape {np.asarray(raw_prediction).shape}.")
#         best_index = int(np.argmax(scores))
#         best_confidence = float(scores[best_index])

#         # Do not reveal a breed prediction when confidence is 75% or lower.
#         # A value strictly greater than the threshold is required.
#         if best_confidence <= CONFIDENCE_THRESHOLD:
#             return {
#                 "success": False,
#                 "message": (
#                     "Unable to identify the breed confidently. Please upload a clear, "
#                     "correct image of a cow or buffalo and try again."
#                 ),
#             }

#         ranked_indices = np.argsort(scores)[::-1]
#         return {
#             "success": True,
#             "predicted_breed": CLASS_NAMES[best_index],
#             "confidence": round(best_confidence, 6),
#             "confidence_percent": round(best_confidence * 100, 2),
#             "top_predictions": [
#                 {
#                     "breed": CLASS_NAMES[int(index)],
#                     "confidence": round(float(scores[index]), 6),
#                     "confidence_percent": round(float(scores[index]) * 100, 2),
#                 }
#                 for index in ranked_indices[:3]
#             ],
#         }
#     except HTTPException:
#         raise
#     except Exception as exc:
#         logger.exception("Model inference failed")
#         raise HTTPException(status_code=500, detail="Prediction failed. Check server logs.") from exc


# @app.post("/predict", tags=["Prediction"])
# async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
#     """Predict the breed from one uploaded JPG, PNG, or WEBP image."""
#     if file.content_type and not file.content_type.startswith("image/"):
#         raise HTTPException(status_code=415, detail="Upload an image file (JPG, PNG, or WEBP).")

#     try:
#         content = await file.read(MAX_UPLOAD_BYTES + 1)
#     finally:
#         await file.close()

#     if not content:
#         raise HTTPException(status_code=400, detail="The uploaded file is empty.")
#     if len(content) > MAX_UPLOAD_BYTES:
#         raise HTTPException(
#             status_code=413,
#             detail=f"Image is too large. Maximum upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
#         )

#     result = await run_in_threadpool(_predict_image, content)
#     result["filename"] = file.filename or "uploaded-image"
#     return result
