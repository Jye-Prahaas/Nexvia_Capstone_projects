import React, { useEffect, useState } from "react";

import {
  DEFAULT_API,
  getHealth,
  predictScan,
  predictDicomSeries,
  sliceUrl,
  maskUrl,
  gradcamUrl,
  getAudit,
  verifyAudit,
  getOrthancResults
} from "./api.js";

import {
  DisclaimerBanner,
  ConfidenceMeter,
  ReviewPanel,
  AuditLog
} from "./components.jsx";


function Stat({ label, value, tone }) {
  return (
    <div className="row">
      <span>{label}</span>
      <span className={tone}>{value}</span>
    </div>
  );
}


export default function App() {

  const [api, setApi] = useState(DEFAULT_API);
  const [health, setHealth] = useState(null);

  const [file, setFile] = useState(null);
  const [dicomFile, setDicomFile] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [result, setResult] = useState(null);
  const [z, setZ] = useState(0);

  const [showMask, setShowMask] = useState(true);
  const [showGradcam, setShowGradcam] = useState(false);

  const [reviewer, setReviewer] = useState("");
  const [decided, setDecided] = useState(null);

  const [audit, setAudit] = useState([]);
  const [integrity, setIntegrity] = useState(null);

  const [orthancData, setOrthancData] = useState({
    total: 0,
    results: {}
  });

  const [orthancLoading, setOrthancLoading] = useState(false);
  const [orthancError, setOrthancError] = useState("");


  const refreshAudit = () =>
    getAudit(api)
      .then(setAudit)
      .catch(() => setAudit([]));


  const refreshOrthanc = async () => {

    setOrthancLoading(true);
    setOrthancError("");

    try {

      const data = await getOrthancResults(api);
      setOrthancData(data);

    } catch (error) {

      setOrthancError("Could not load Orthanc results.");

    } finally {

      setOrthancLoading(false);

    }

  };


  const runVerify = () =>
    verifyAudit(api)
      .then(setIntegrity)
      .catch(() => setIntegrity(null));


  // Backend connection and periodic Orthanc refresh.
  useEffect(() => {

    let cancelled = false;

    getHealth(api)
      .then((h) => !cancelled && setHealth(h))
      .catch(() => !cancelled && setHealth(null));

    refreshAudit();
    refreshOrthanc();

    const orthancTimer = setInterval(() => {

      if (!cancelled) {
        refreshOrthanc();
      }

    }, 15000);

    return () => {

      cancelled = true;
      clearInterval(orthancTimer);

    };

  }, [api]);


  // Keyboard navigation through CT slices.
  useEffect(() => {

    if (!result) return;

    const onKey = (e) => {

      if (e.key === "ArrowUp") {

        e.preventDefault();

        setZ((v) =>
          Math.min(result.num_slices - 1, v + 1)
        );

      }

      if (e.key === "ArrowDown") {

        e.preventDefault();

        setZ((v) =>
          Math.max(0, v - 1)
        );

      }

    };

    window.addEventListener("keydown", onKey);

    return () =>
      window.removeEventListener("keydown", onKey);

  }, [result]);


  function handleResult(data) {

    setResult(data);
    setZ(data.best_slice ?? 0);

    setDecided(null);
    setReviewer("");

    setShowMask(true);
    setShowGradcam(false);

  }


  // NIfTI prediction.
  async function run() {

    if (!file) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {

      const data = await predictScan(api, file);

      handleResult(data);

    } catch (e) {

      setError(
        e.message || "NIfTI processing failed."
      );

    } finally {

      setLoading(false);

    }

  }


  // DICOM series prediction.
  async function runDicom() {

    if (!dicomFile) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {

      const data = await predictDicomSeries(
        api,
        dicomFile
      );

      handleResult(data);

    } catch (e) {

      setError(
        e.message || "DICOM processing failed."
      );

    } finally {

      setLoading(false);

    }

  }


  const inTumour =
    result?.tumour_detected &&
    z >= result.first_tumour_slice &&
    z <= result.last_tumour_slice;


  return (

    <div className="wrap">

      <DisclaimerBanner />

      <h1>Lung Tumour Segmentation Viewer</h1>

      <div className="sub">
        3D U-Net · NIfTI and DICOM CT analysis ·
        3D Grad-CAM explainability
      </div>


      <div className="grid">

        {/* CT VIEWER */}

        <div className="panel">

          <div className="viewer">

            {result ? (

              <>

                {/* Original CT slice */}

                <img
                  alt="CT slice"
                  src={sliceUrl(api, result.scan_id, z)}
                />


                {/* 3D Grad-CAM heatmap */}

                {showGradcam && (

                  <img
                    key={`gradcam-${result.scan_id}-${z}`}
                    alt="3D CT Grad-CAM heatmap"
                    src={gradcamUrl(api, result.scan_id, z)}
                    style={{
                      position: "absolute",
                      inset: 0,
                      width: "100%",
                      height: "100%",
                      objectFit: "contain",
                      pointerEvents: "none",
                      zIndex: 2
                    }}
                  />

                )}


                {/* Tumour segmentation mask */}

                {showMask && (

                  <img
                    alt="Predicted tumour segmentation"
                    src={maskUrl(api, result.scan_id, z)}
                  />

                )}


                <div className="badge">

                  Slice {z + 1} / {result.num_slices}

                  {inTumour ? " · tumour" : ""}

                </div>

              </>

            ) : (

              <div className="empty">

                {loading

                  ? "Running the model… this can take several minutes on CPU"

                  : "Upload a CT scan to begin analysis"

                }

              </div>

            )}

          </div>


          {/* CT slice navigation */}

          {result && (

            <>

              <input
                type="range"
                min="0"
                max={result.num_slices - 1}
                value={z}
                onChange={(e) =>
                  setZ(Number(e.target.value))
                }
              />


              <div className="controls">

                <label>

                  <input
                    type="checkbox"
                    checked={showMask}
                    onChange={(e) =>
                      setShowMask(e.target.checked)
                    }
                  />

                  {" "}Show tumour overlay

                </label>


                <label>

                  <input
                    type="checkbox"
                    checked={showGradcam}
                    onChange={(e) =>
                      setShowGradcam(e.target.checked)
                    }
                  />

                  {" "}Show Grad-CAM heatmap

                </label>


                {result.tumour_detected && (

                  <button
                    className="ghost"
                    onClick={() =>
                      setZ(result.best_slice)
                    }
                  >
                    Jump to largest slice
                  </button>

                )}


                <span style={{ color: "var(--muted)" }}>
                  ↑ ↓ keys scroll
                </span>

              </div>


              {showGradcam && (

                <div className="note">

                  Grad-CAM highlights regions that
                  influenced the model's tumour-class
                  prediction. The current implementation
                  generates a localized heatmap around
                  the highest predicted probability region.

                </div>

              )}

            </>

          )}

        </div>


        {/* UPLOAD AND RESULTS PANEL */}

        <div className="panel">

          <h3>CT Scan Upload</h3>


          {/* NIFTI UPLOAD */}

          <div className="note">
            Option 1: Upload NIfTI CT scan
          </div>

          <label className="file">

            <input
              type="file"
              accept=".nii,.nii.gz"
              style={{ display: "none" }}
              onChange={(e) =>
                setFile(e.target.files[0] || null)
              }
            />

            {file ? file.name : "Choose NIfTI file"}

          </label>


          <button
            onClick={run}
            disabled={!file || loading}
            style={{
              width: "100%",
              marginBottom: 20
            }}
          >

            {loading
              ? "Processing..."
              : "Run NIfTI segmentation"
            }

          </button>


          <hr />


          {/* DICOM ZIP UPLOAD */}

          <div className="note">
            Option 2: Upload complete DICOM CT series as ZIP
          </div>

          <label className="file">

            <input
              type="file"
              accept=".zip,application/zip"
              style={{ display: "none" }}
              onChange={(e) =>
                setDicomFile(e.target.files[0] || null)
              }
            />

            {dicomFile
              ? dicomFile.name
              : "Choose DICOM ZIP file"
            }

          </label>


          <button
            onClick={runDicom}
            disabled={!dicomFile || loading}
            style={{ width: "100%" }}
          >

            {loading
              ? "Processing..."
              : "Run DICOM segmentation"
            }

          </button>


          {error && (

            <div className="status err">
              {error}
            </div>

          )}


          {/* API STATUS */}

          <div className="status">

            API URL

            <input
              className="api-input"
              value={api}
              onChange={(e) =>
                setApi(e.target.value)
              }
            />

            <div
              style={{ marginTop: 4 }}
              className={
                health?.classifier_loaded &&
                health?.segmentation_loaded
                  ? "good"
                  : ""
              }
            >

              {health

                ? health.classifier_loaded &&
                  health.segmentation_loaded

                  ? `Connected · Both models loaded · ${health.device}`

                  : `Connected · Classifier: ${
                      health.classifier_loaded
                        ? "Loaded"
                        : "Not loaded"
                    } · Segmentation: ${
                      health.segmentation_loaded
                        ? "Loaded"
                        : "Not loaded"
                    }`

                : "Cannot reach API"

              }

            </div>

          </div>


          {/* SEGMENTATION RESULTS */}

          {result && (

            <div style={{ marginTop: 14 }}>

              <Stat
                label="Tumour detected"
                value={
                  result.tumour_detected
                    ? "Yes"
                    : "No"
                }
                tone={
                  result.tumour_detected
                    ? "err"
                    : "good"
                }
              />

              <ConfidenceMeter
                value={result.confidence}
              />

              <Stat
                label="Predicted volume"
                value={`${result.tumour_volume_ml} mL`}
              />

              <Stat
                label="Voxels"
                value={result.tumour_voxels.toLocaleString()}
              />

              <Stat
                label="Tumour slices"
                value={
                  result.tumour_detected
                    ? `${result.first_tumour_slice + 1}–${result.last_tumour_slice + 1}`
                    : "—"
                }
              />

              <Stat
                label="Volume shape"
                value={result.shape.join(" × ")}
              />

              <Stat
                label="Input"
                value={result.filename}
              />

            </div>

          )}


          {/* REVIEW PANEL */}

          {result && (

            <ReviewPanel
              api={api}
              scan={result}
              reviewer={reviewer}
              setReviewer={setReviewer}
              decided={decided}
              onDecided={(rec) => {
                setDecided(rec);
                refreshAudit();
              }}
            />

          )}


          <div className="note">

            Educational research prototype only.
            Predictions are not medical diagnoses.
            Results require independent validation.

          </div>

        </div>

      </div>


      {/* AUDIT LOG */}

      <AuditLog
        entries={audit}
        integrity={integrity}
        onRefresh={refreshAudit}
        onVerify={runVerify}
      />


      {/* AUTOMATIC ORTHANC RESULTS */}

      <div className="panel" style={{ marginTop: 24 }}>

        <h2>Automatically Processed Orthanc Studies</h2>

        <div className="sub">
          Results generated automatically from Orthanc DICOM studies.
        </div>


        <div className="row">

          <span>Total processed studies</span>

          <strong>{orthancData.total}</strong>

        </div>


        <button
          className="ghost"
          onClick={refreshOrthanc}
          disabled={orthancLoading}
          style={{ marginTop: 12 }}
        >

          {orthancLoading
            ? "Refreshing..."
            : "Refresh Orthanc Results"
          }

        </button>


        {orthancError && (

          <div className="status err">
            {orthancError}
          </div>

        )}


        {orthancData.total === 0 ? (

          <div className="note">
            No automatically processed studies available yet.
          </div>

        ) : (

          Object.entries(orthancData.results).map(
            ([studyId, item]) => (

              <div
                key={studyId}
                className="panel"
                style={{ marginTop: 16 }}
              >

                <h3>Orthanc Study</h3>

                <Stat
                  label="Scan ID"
                  value={item.scan_id}
                />

                <Stat
                  label="Tumour detection threshold"
                  value={
                    item.tumour_detected
                      ? "Region detected"
                      : "No region above threshold"
                  }
                />

                <Stat
                  label="Predicted volume"
                  value={`${item.tumour_volume_ml} mL`}
                />

                <Stat
                  label="Predicted voxels"
                  value={item.tumour_voxels.toLocaleString()}
                />

                <Stat
                  label="Confidence"
                  value={
                    item.confidence !== null
                      ? `${(
                          item.confidence * 100
                        ).toFixed(1)}%`
                      : "Unavailable"
                  }
                />

                <Stat
                  label="Processing device"
                  value={item.device}
                />

                <Stat
                  label="Processed at"
                  value={item.processed_at}
                />

              </div>

            )
          )

        )}


        <div className="note">

          Automated predictions are research outputs,
          not medical diagnoses. Independent validation
          is required.

        </div>

      </div>

    </div>

  );

}