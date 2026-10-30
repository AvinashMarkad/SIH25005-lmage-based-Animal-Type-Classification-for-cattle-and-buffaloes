# Cow & Buffalo Breed Classifier — FastAPI Backend

A FastAPI API for the supplied Keras model. It accepts a photo and predicts one of these six trained classes:

- Gir
- Holstein_Friesian
- Jaffrabadi
- Jersey
- Murrah
- Sahiwal

The uploaded model is included as `cow_breed_classifier.keras`.

## 1. Install Python

Use Python 3.11 or 3.12 for a straightforward TensorFlow setup. Make sure Python is added to PATH.

Check your installation:

```powershell
python --version
```

## 2. Open this folder in the terminal

In VS Code, open the `cow_breed_fastapi_backend` folder, then choose **Terminal → New Terminal**.

## 3. Create and activate a virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the current terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**Windows note:** If TensorFlow installation fails on your Windows/Python combination, use Python 3.11 in a fresh virtual environment or use WSL2. Do not install packages into a different Python environment from the one used to run Uvicorn.

## 4. Start the backend

```powershell
uvicorn main:app --reload
```

Open these URLs in your browser:

- API status: http://127.0.0.1:8000/
- Health check: http://127.0.0.1:8000/health
- Interactive API docs: http://127.0.0.1:8000/docs

The model loads once when the server starts. The first startup can take a little time.

## 5. Test a prediction

### In Swagger UI

1. Open http://127.0.0.1:8000/docs
2. Expand **POST /predict** and click **Try it out**.
3. Select a JPG, PNG, or WEBP image and click **Execute**.

### With PowerShell/cURL

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" -F "file=@C:\path\to\cow.jpg"
```

Example response shape (the scores depend on the uploaded image):

```json
{
  "predicted_breed": "Gir",
  "confidence": 0.93421,
  "confidence_percent": 93.42,
  "top_predictions": [
    {"breed": "Gir", "confidence": 0.93421, "confidence_percent": 93.42},
    {"breed": "Sahiwal", "confidence": 0.041, "confidence_percent": 4.1},
    {"breed": "Jersey", "confidence": 0.015, "confidence_percent": 1.5}
  ],
  "filename": "cow.jpg"
}
```

The example scores above are illustrative, not a real prediction.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API welcome message and endpoint links |
| GET | `/health` | Model-loaded status and configured classes |
| POST | `/predict` | Accepts multipart form field `file`; returns breed and confidence |
| GET | `/docs` | Interactive Swagger documentation |

## Connect a frontend

The development CORS setting allows requests from any origin. Before deployment, set `CORS_ORIGINS` to the exact frontend origin(s), separated by commas, for example:

```powershell
$env:CORS_ORIGINS="http://localhost:5500,http://127.0.0.1:5500"
uvicorn main:app --reload
```

## Model details and limitations

- The API follows the supplied prediction script: resize to `224×224`, convert to RGB, scale pixel values to `0–1`, then call the Keras model.
- Class labels and order follow the supplied `predict(5).py` script. If the model was trained with a different class-index order, update `CLASS_NAMES` to the exact order printed by `train_model(1).py`.
- Confidence is the model's softmax score, not a guarantee that the prediction is correct.
- Only JPG/JPEG, PNG and WEBP images are accepted; the default maximum upload size is 10 MB.
