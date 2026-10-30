# BreedKind — Cattle & Buffalo Breed Identifier

Responsive HTML/CSS/JavaScript frontend for the existing FastAPI endpoint `POST /predict`. It uses Bootstrap 5 for the base layout utilities and custom CSS for the Organic sand, moss, and fired-clay visual style.

## Features

- Upload or drag-and-drop a JPG, PNG, or WEBP image (10 MB maximum).
- Sends the image as `multipart/form-data` using the field name `file`, matching the shown `/predict` endpoint.
- Previews the selected photo before submission.
- Shows breed details and top predictions only if confidence is **strictly greater than 75%**.
- If confidence is 75% or below, shows: **“Upload a clear or correct image.”** and does not show the predicted breed.
- Handles API errors, keyboard use, visible focus, responsive layouts, and reduced-motion preferences.

## Connect it to your existing FastAPI app

The easiest setup is to serve the frontend from the same FastAPI process. That keeps the browser request same-origin and avoids CORS problems.

1. Copy the `frontend` folder next to your existing `main.py`.
2. In `main.py`, add these imports (if you do not already have them):

   ```python
   from pathlib import Path
   from fastapi.responses import FileResponse
   from fastapi.staticfiles import StaticFiles
   ```

3. After creating your existing `app = FastAPI(...)`, add this mount and route. If your app already defines a `/` route, update that route to return the `index.html` file rather than creating a duplicate route.

   ```python
   BASE_DIR = Path(__file__).resolve().parent
   FRONTEND_DIR = BASE_DIR / "frontend"

   app.mount(
       "/frontend-assets",
       StaticFiles(directory=str(FRONTEND_DIR)),
       name="frontend-assets",
   )

   @app.get("/", include_in_schema=False)
   async def frontend_home():
       return FileResponse(str(FRONTEND_DIR / "index.html"))
   ```

4. Restart FastAPI, for example:

   ```bash
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```

5. Open **http://127.0.0.1:8000/**. Your API documentation remains at **http://127.0.0.1:8000/docs** and predictions still go to **POST http://127.0.0.1:8000/predict**.

> If your Python module is not named `main.py`, replace `main:app` with your module name. If your app uses a different port, the frontend automatically targets `/predict` on the same origin.

## Expected response

The frontend is compatible with a response shaped like this:

```json
{
  "success": true,
  "predicted_breed": "Gir",
  "confidence": 0.791178,
  "confidence_percent": 79.12,
  "top_predictions": [
    {"breed": "Gir", "confidence": 0.791178, "confidence_percent": 79.12},
    {"breed": "Sahiwal", "confidence": 0.208798, "confidence_percent": 20.88}
  ],
  "filename": "animal.jpg"
}
```

The confidence gate is enforced in the frontend for display purposes. For consistent application behavior, keep the same 75% condition in the FastAPI backend as well.

## Notes

- Bootstrap and Google Fonts are loaded from CDNs, so those assets need an internet connection. The page itself and prediction API run locally.
- The frontend expects the upload field name to be `file`, matching the endpoint shown in the screenshots.
