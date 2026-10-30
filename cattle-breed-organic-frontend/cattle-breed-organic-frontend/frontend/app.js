(() => {
  "use strict";

  // Served by the same FastAPI application, so no cross-origin request is needed.
  const API_URL = `${window.location.origin}/predict`;
  const CONFIDENCE_THRESHOLD = 75;
  const MAX_FILE_SIZE = 10 * 1024 * 1024;
  const ALLOWED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);

  const input = document.getElementById("image-input");
  const dropZone = document.getElementById("drop-zone");
  const chooseButton = document.getElementById("choose-button");
  const predictButton = document.getElementById("predict-button");
  const removeButton = document.getElementById("remove-image");
  const resetButton = document.getElementById("reset-button");
  const previewPanel = document.getElementById("preview-panel");
  const preview = document.getElementById("image-preview");
  const fileNameLabel = document.getElementById("selected-filename");
  const message = document.getElementById("form-message");
  const resultSection = document.getElementById("result-section");
  const resultContent = document.getElementById("result-content");
  const resultStatus = document.getElementById("result-status");

  let selectedFile = null;
  let previewUrl = null;
  let requestInProgress = false;

  function setMessage(text, type = "") {
    message.textContent = text;
    message.className = `form-message${type ? ` ${type}` : ""}`;
  }

  function setBusy(isBusy) {
    requestInProgress = isBusy;
    predictButton.disabled = isBusy || !selectedFile;
    predictButton.setAttribute("aria-busy", String(isBusy));
    predictButton.innerHTML = isBusy
      ? '<span class="button-spinner" aria-hidden="true"></span> Reading image…'
      : 'Identify breed <span aria-hidden="true">↗</span>';
  }

  function revokePreviewUrl() {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      previewUrl = null;
    }
  }

  function setFile(file) {
    if (!file) return;
    if (!ALLOWED_TYPES.has(file.type)) {
      setMessage("Please choose a JPG, PNG or WEBP image.", "error");
      return;
    }
    if (file.size > MAX_FILE_SIZE) {
      setMessage("This image is larger than 10 MB. Choose a smaller image and try again.", "error");
      return;
    }
    if (!file.size) {
      setMessage("This file appears to be empty. Please choose another image.", "error");
      return;
    }

    revokePreviewUrl();
    selectedFile = file;
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    preview.alt = `Preview of ${file.name}`;
    fileNameLabel.textContent = file.name;
    dropZone.hidden = true;
    previewPanel.hidden = false;
    predictButton.disabled = false;
    resultSection.hidden = true;
    resultContent.replaceChildren();
    setMessage("Image ready. Check the preview, then identify the breed.", "success");
  }

  function resetForm({ keepMessage = false } = {}) {
    if (requestInProgress) return;
    selectedFile = null;
    revokePreviewUrl();
    input.value = "";
    preview.removeAttribute("src");
    previewPanel.hidden = true;
    dropZone.hidden = false;
    predictButton.disabled = true;
    resultSection.hidden = true;
    resultContent.replaceChildren();
    if (!keepMessage) setMessage("");
  }

  function getConfidencePercent(data) {
    const pct = Number(data?.confidence_percent);
    if (Number.isFinite(pct)) return pct;
    const confidence = Number(data?.confidence);
    if (Number.isFinite(confidence)) return confidence <= 1 ? confidence * 100 : confidence;
    const firstTop = Array.isArray(data?.top_predictions) ? data.top_predictions[0] : null;
    const topPct = Number(firstTop?.confidence_percent);
    if (Number.isFinite(topPct)) return topPct;
    const topConfidence = Number(firstTop?.confidence);
    if (Number.isFinite(topConfidence)) return topConfidence <= 1 ? topConfidence * 100 : topConfidence;
    return 0;
  }

  function formatPercent(value) {
    return `${Math.max(0, Math.min(100, value)).toFixed(2).replace(/\.00$/, "").replace(/(\.\d)0$/, "$1")}%`;
  }

  function normalizePredictions(data, mainConfidence) {
    const predictions = Array.isArray(data?.top_predictions) ? data.top_predictions : [];
    if (predictions.length) {
      return predictions
        .map((item) => {
          const raw = Number(item?.confidence_percent);
          const confidence = Number(item?.confidence);
          const percent = Number.isFinite(raw) ? raw : Number.isFinite(confidence) ? (confidence <= 1 ? confidence * 100 : confidence) : 0;
          return { breed: String(item?.breed || "Unknown breed"), percent };
        })
        .filter((item) => item.breed && Number.isFinite(item.percent))
        .slice(0, 5);
    }
    return [{ breed: String(data?.predicted_breed || "Predicted breed"), percent: mainConfidence }];
  }

  function showLowConfidence(percent) {
    resultStatus.textContent = "NEEDS A CLEARER IMAGE";
    resultContent.innerHTML = `
      <div class="low-confidence">
        <div class="low-confidence-icon" aria-hidden="true">↻</div>
        <div>
          <p class="result-kicker">Confidence at or below ${CONFIDENCE_THRESHOLD}%</p>
          <h3 id="result-title">Upload a clear or correct image.</h3>
          <p>The model isn't confident enough to show a breed prediction (${formatPercent(percent)}). Try a well-lit photo where the animal is in focus and clearly visible.</p>
          <a href="#identify" class="text-link">Choose another image <span aria-hidden="true">↗</span></a>
        </div>
      </div>`;
    resultSection.hidden = false;
    resultSection.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  }

  function showSuccessfulPrediction(data, confidencePercent) {
    const breed = String(data.predicted_breed || data.breed || "Unknown breed");
    const predictions = normalizePredictions(data, confidencePercent);
    const hasImagePreview = Boolean(previewUrl);
    const safePreviewMarkup = hasImagePreview
      ? '<div><img class="result-image-thumb" id="result-image-thumb" alt="The image used for this breed prediction"></div>'
      : "";
    const predictionsMarkup = predictions.map((prediction) => `
      <div class="prediction-row">
        <strong>${escapeHtml(prediction.breed)}</strong>
        <span>${formatPercent(prediction.percent)}</span>
      </div>`).join("");

    resultStatus.textContent = "CONFIDENT RESULT";
    resultContent.innerHTML = `
      <div class="result-success-grid">
        <div>
          <p class="result-kicker">Most likely breed</p>
          <h3 class="predicted-breed" id="result-title">${escapeHtml(breed)}</h3>
          <p class="result-footnote">The model's top prediction cleared the ${CONFIDENCE_THRESHOLD}% confidence threshold.</p>
          <div class="confidence-row"><span>Model confidence</span><strong>${formatPercent(confidencePercent)}</strong></div>
          <div class="confidence-track" role="progressbar" aria-label="Model confidence" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(confidencePercent)}"><div class="confidence-fill" style="width:${Math.max(0, Math.min(100, confidencePercent))}%"></div></div>
        </div>
        ${safePreviewMarkup}
      </div>
      <div class="top-predictions-wrap" style="margin-top:28px">
        <p class="result-kicker">Top model predictions</p>
        <div class="prediction-list">${predictionsMarkup}</div>
      </div>
      <p class="result-footnote">This result is generated by your image classifier. Treat it as an estimate; image quality and similarity between breeds can affect confidence.</p>`;
    if (hasImagePreview) {
      const resultThumb = document.getElementById("result-image-thumb");
      if (resultThumb) resultThumb.src = previewUrl;
    }
    resultSection.hidden = false;
    resultSection.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  }

  function showApiError(text) {
    resultStatus.textContent = "REQUEST COULD NOT COMPLETE";
    resultContent.innerHTML = `
      <div class="low-confidence">
        <div class="low-confidence-icon" aria-hidden="true">!</div>
        <div>
          <p class="result-kicker">Connection or prediction issue</p>
          <h3 id="result-title">We couldn't read this image.</h3>
          <p>Check that your FastAPI server is running and the <code>POST /predict</code> route is available, then try again.</p>
          <p class="error-detail">${escapeHtml(text)}</p>
          <a href="#identify" class="text-link">Return to image upload <span aria-hidden="true">↗</span></a>
        </div>
      </div>`;
    resultSection.hidden = false;
    resultSection.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (character) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    })[character]);
  }

  async function predict() {
    if (!selectedFile || requestInProgress) return;
    setBusy(true);
    resultSection.hidden = true;
    setMessage("Sending image to the classifier…", "loading");

    const formData = new FormData();
    formData.append("file", selectedFile, selectedFile.name);

    try {
      const response = await fetch(API_URL, { method: "POST", body: formData, headers: { Accept: "application/json" } });
      let data;
      try {
        data = await response.json();
      } catch {
        throw new Error(`The API returned an unreadable response (HTTP ${response.status}).`);
      }
      if (!response.ok) {
        const detail = typeof data?.detail === "string" ? data.detail : `Request failed with HTTP ${response.status}.`;
        throw new Error(detail);
      }

      const confidencePercent = getConfidencePercent(data);
      const modelSaysSuccess = data?.success !== false;
      if (modelSaysSuccess && confidencePercent > CONFIDENCE_THRESHOLD && (data?.predicted_breed || data?.breed)) {
        showSuccessfulPrediction(data, confidencePercent);
        setMessage("Prediction complete.", "success");
      } else {
        showLowConfidence(confidencePercent);
        setMessage("The image did not pass the confidence threshold.", "error");
      }
    } catch (error) {
      const details = error instanceof Error ? error.message : "An unexpected error occurred.";
      const isNetworkError = error instanceof TypeError;
      showApiError(isNetworkError ? "Unable to connect. If the page is not served by FastAPI, check the CORS setup in README.md." : details);
      setMessage("Prediction failed. Check the API server and try again.", "error");
    } finally {
      setBusy(false);
    }
  }

  chooseButton.addEventListener("click", (event) => {
    event.stopPropagation();
    input.click();
  });
  dropZone.addEventListener("click", (event) => {
    if (event.target.closest("button")) return;
    input.click();
  });
  dropZone.addEventListener("keydown", (event) => {
    if (event.target !== dropZone) return;
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.click();
    }
  });
  input.addEventListener("change", () => {
    setFile(input.files && input.files[0]);
    input.value = "";
  });
  ["dragenter", "dragover"].forEach((eventName) => dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragging");
  }));
  ["dragleave", "drop"].forEach((eventName) => dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragging");
  }));
  dropZone.addEventListener("drop", (event) => {
    const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
    setFile(file);
  });
  predictButton.addEventListener("click", predict);
  removeButton.addEventListener("click", () => resetForm());
  resetButton.addEventListener("click", () => resetForm());
  window.addEventListener("beforeunload", revokePreviewUrl);
})();
