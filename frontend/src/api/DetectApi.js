const DEFAULT_API_BASE_URL = "http://127.0.0.1:5000";
const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL
).replace(/\/$/, "");

const buildApiUrl = (path) => `${API_BASE_URL}${path}`;

const DETECT_TIMEOUT_MS = 60000;
const MAX_IMAGE_SIDE = 1280;
const JPEG_QUALITY = 0.86;

const readAsDataUrl = (file) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Failed to read image file"));
    reader.readAsDataURL(file);
  });

const loadImage = (src) =>
  new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Failed to decode image file"));
    image.src = src;
  });

const stripDataUrlPrefix = (dataUrl) => dataUrl.split(",")[1] || "";

/**
 * Converts an image File object to a compact base64 JPEG payload.
 */
const toBase64 = async (file) => {
  const dataUrl = await readAsDataUrl(file);
  const image = await loadImage(dataUrl);
  const longestSide = Math.max(image.width, image.height);
  const scale = Math.min(1, MAX_IMAGE_SIDE / longestSide);
  const width = Math.max(1, Math.round(image.width * scale));
  const height = Math.max(1, Math.round(image.height * scale));

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.drawImage(image, 0, 0, width, height);

  return stripDataUrlPrefix(canvas.toDataURL("image/jpeg", JPEG_QUALITY));
};

/**
 * Sends the image to the Flask backend and returns the detection result.
 * @param {File} file - The image file selected by the user
 * @returns {Promise<{ insect_name: string, confidence: number }>}
 */
export async function detectInsect(file) {
  // Step 1: convert image to base64
  const image_base64 = await toBase64(file);
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), DETECT_TIMEOUT_MS);

  try {
    // Step 2: POST to Flask detection endpoint
    const response = await fetch(buildApiUrl("/api/detect"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: image_base64 }),
      signal: controller.signal,
    });

    // Step 3: handle non-ok HTTP responses
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || `Server error: ${response.status}`);
    }

    // Step 4: return the parsed result
    // Expected response shape: { insect_name: "...", confidence: 0.95 }
    const data = await response.json();
    return data;
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error(
        "Detection timed out. Please try a clearer or smaller image.",
        { cause: error }
      );
    }

    throw error;
  } finally {
    window.clearTimeout(timeoutId);
  }
}

export async function getAiStatus() {
  const response = await fetch(buildApiUrl("/api/ai/status"));

  if (!response.ok) {
    throw new Error(`Server error: ${response.status}`);
  }

  return response.json();
}

export async function getAiContract() {
  const response = await fetch(buildApiUrl("/api/ai/contract"));

  if (!response.ok) {
    throw new Error(`Server error: ${response.status}`);
  }

  return response.json();
}
