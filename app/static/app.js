const form = document.querySelector("#upload-form");
const input = document.querySelector("#image-input");
const dropZone = document.querySelector("#drop-zone");
const dropCopy = document.querySelector("#drop-copy");
const preview = document.querySelector("#preview");
const submitButton = document.querySelector("#submit-button");
const statusBox = document.querySelector("#status");
const resultBox = document.querySelector("#result");
const resultLabel = document.querySelector("#result-label");
const resultConfidence = document.querySelector("#result-confidence");
const probabilityList = document.querySelector("#probability-list");
const latencyValue = document.querySelector("#latency-value");
const deviceValue = document.querySelector("#device-value");
const heatmap = document.querySelector("#heatmap");
const neighborList = document.querySelector("#neighbor-list");

let selectedFile = null;
let previewUrl = null;

const labels = {
  bicycle: "Bicycle",
  bus: "Bus",
  car: "Car",
  motorcycle: "Motorcycle",
  truck: "Truck",
};

function displayLabel(label) {
  return labels[label.toLowerCase()] || label;
}

function selectFile(file) {
  if (!file || !["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
    showError("Select an image in JPG, PNG, or WebP format.");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showError("The file is too large. The maximum size is 10 MB.");
    return;
  }

  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  preview.hidden = false;
  dropCopy.hidden = true;
  submitButton.disabled = false;
  statusBox.textContent = file.name;
  statusBox.classList.remove("error");
  resultBox.hidden = true;
}

function showError(message) {
  statusBox.textContent = message;
  statusBox.classList.add("error");
}

input.addEventListener("change", () => selectFile(input.files[0]));

for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
}

dropZone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedFile) return;

  submitButton.disabled = true;
  submitButton.textContent = "Analyzing…";
  statusBox.textContent = "The model is processing the image";
  statusBox.classList.remove("error");
  resultBox.hidden = true;

  const body = new FormData();
  body.append("file", selectedFile);

  try {
    const response = await fetch("/api/predict", { method: "POST", body });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "The image could not be classified.");

    resultLabel.textContent = displayLabel(payload.label);
    resultConfidence.textContent = `${Math.round(payload.confidence * 100)}%`;
    probabilityList.replaceChildren(
      ...payload.probabilities.map((item) => {
        const row = document.createElement("div");
        row.className = "probability-row";
        row.innerHTML = `
          <span>${displayLabel(item.label)}</span>
          <span class="bar"><i style="width: ${item.confidence * 100}%"></i></span>
          <span class="probability-value">${Math.round(item.confidence * 100)}%</span>
        `;
        return row;
      }),
    );
    latencyValue.textContent = payload.latency_ms.toFixed(1);
    deviceValue.textContent = payload.device;
    heatmap.src = payload.heatmap;
    neighborList.replaceChildren(
      ...payload.neighbors.map((item) => {
        const figure = document.createElement("figure");
        figure.className = "neighbor";
        const image = document.createElement("img");
        image.src = item.url;
        image.alt = `Similar example: ${displayLabel(item.label)}`;
        const caption = document.createElement("figcaption");
        caption.textContent = `${displayLabel(item.label)} · ${Math.round(item.similarity * 100)}%`;
        figure.append(image, caption);
        return figure;
      }),
    );
    statusBox.textContent = "Complete";
    resultBox.hidden = false;
  } catch (error) {
    showError(error.message);
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Classify vehicle";
  }
});

