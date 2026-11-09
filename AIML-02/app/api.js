export const DEFAULT_API =
  import.meta.env.VITE_API_URL || "http://localhost:8000";

async function handleError(res) {
  const body = await res.json().catch(() => ({}));
  throw new Error(body.detail || `Server error ${res.status}`);
}

export async function getHealth(api) {
  const res = await fetch(`${api}/health`);

  if (!res.ok) throw new Error("API unreachable");

  return res.json();
}

export async function predictScan(api, file) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${api}/predict`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) await handleError(res);

  return res.json();
}

export async function predictDicomSeries(api, file) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${api}/predict/dicom-series`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) await handleError(res);

  return res.json();
}

// Grad-CAM chest X-ray prediction
export async function predictGradcam(api, file) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${api}/predict/gradcam`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) await handleError(res);

  const imageBlob = await res.blob();

  return {
    imageUrl: URL.createObjectURL(imageBlob),
    prediction: res.headers.get("x-prediction"),
    probability: res.headers.get("x-probability"),
  };
}

export const sliceUrl = (api, id, z) =>
  `${api}/slice/${id}/${z}`;

export const maskUrl = (api, id, z) =>
  `${api}/mask/${id}/${z}`;

export const gradcamUrl = (api, id, z) => `${api}/gradcam/${id}/${z}`;

export async function submitDecision(api, payload) {
  const res = await fetch(`${api}/decision`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const d = body.detail;

    throw new Error(
      typeof d === "string" ? d : "Could not save decision"
    );
  }

  return res.json();
}

export async function getAudit(api, limit = 50) {
  const res = await fetch(`${api}/audit?limit=${limit}`);

  if (!res.ok) throw new Error("Could not load audit log");

  return res.json();
}

export async function verifyAudit(api) {
  const res = await fetch(`${api}/audit/verify`);

  if (!res.ok) throw new Error("Could not verify audit log");

  return res.json();
}

export async function getOrthancResults(api) {
  const res = await fetch(`${api}/orthanc/results`);

  if (!res.ok) {
    throw new Error("Could not load Orthanc results");
  }

  return res.json();
}