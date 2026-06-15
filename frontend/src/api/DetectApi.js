const DEFAULT_API_BASE_URL = "http://localhost:5000";
const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL
).replace(/\/$/, "");

const buildApiUrl = (path) => `${API_BASE_URL}${path}`;

/**
 * Converts an image File object to a base64 string (without the prefix).
 */
const toBase64 = (file) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(",")[1]); // strip "data:image/...;base64,"
    reader.onerror = () => reject(new Error("Failed to read image file"));
    reader.readAsDataURL(file);
  });

/**
 * Sends the image to the Flask backend and returns the detection result.
 * @param {File} file - The image file selected by the user
 * @returns {Promise<{ insect_name: string, confidence: number }>}
 */
export async function detectInsect(file) {
  // Step 1: convert image to base64
  const image_base64 = await toBase64(file);

  // Step 2: POST to Flask detection endpoint
  const response = await fetch(buildApiUrl("/api/detect"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: image_base64 }),
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
