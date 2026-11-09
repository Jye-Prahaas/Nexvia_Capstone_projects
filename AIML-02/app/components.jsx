import React, { useState } from "react";
import { submitDecision } from "./api.js";

export function DisclaimerBanner() {
  return (
    <div className="disclaimer" role="alert">
      <strong>FOR EDUCATION ONLY — NOT FOR CLINICAL USE.</strong> This is a student research
      prototype. Its output must not be used for diagnosis, treatment or any patient-care decision.
    </div>
  );
}

export function ConfidenceMeter({ value }) {
  if (value == null) {
    return <div className="row"><span>Model confidence</span><span>— (no tumour predicted)</span></div>;
  }
  const pct = Math.round(value * 100);
  const tone = pct >= 80 ? "good" : pct >= 60 ? "warn" : "err";
  return (
    <div style={{ margin: "10px 0" }}>
      <div className="row" style={{ borderBottom: "none" }}>
        <span>Model confidence</span>
        <span className={tone}>{pct}%</span>
      </div>
      <div className="meter"><div className={`meter-fill ${tone}`} style={{ width: `${pct}%` }} /></div>
      <div className="note" style={{ marginTop: 4 }}>
        Mean model probability inside the predicted region. Not a calibrated probability of being correct.
      </div>
    </div>
  );
}

export function ReviewPanel({ api, scan, reviewer, setReviewer, decided, onDecided }) {
  const [comment, setComment] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  async function decide(decision) {
    if (!reviewer.trim()) { setErr("Enter your name before deciding."); return; }
    setBusy(true); setErr("");
    try {
      const rec = await submitDecision(api, {
        scan_id: scan.scan_id, decision, reviewer: reviewer.trim(), comment,
      });
      onDecided(rec);
    } catch (e) {
      setErr(e.message);
    } finally { setBusy(false); }
  }

  if (decided) {
    return (
      <div className={`decided ${decided.decision}`}>
        <strong>{decided.decision === "approved" ? "Approved" : "Rejected"}</strong> by {decided.reviewer}
        <div className="note">Recorded in the audit log at {decided.timestamp_utc} (entry #{decided.id}).</div>
      </div>
    );
  }

  return (
    <div className="review">
      <div className="review-title">Radiologist review</div>
      <input className="text-input" placeholder="Reviewer name (required)" value={reviewer}
             onChange={(e) => setReviewer(e.target.value)} />
      <textarea className="text-input" rows={2} placeholder="Comment (optional)" value={comment}
                onChange={(e) => setComment(e.target.value)} />
      <div className="btn-row">
        <button className="approve" disabled={busy} onClick={() => decide("approved")}>Approve</button>
        <button className="reject" disabled={busy} onClick={() => decide("rejected")}>Reject</button>
      </div>
      {err && <div className="status err">{err}</div>}
    </div>
  );
}

export function AuditLog({ entries, integrity, onRefresh, onVerify }) {
  return (
    <div className="panel" style={{ marginTop: 16 }}>
      <div className="audit-head">
        <strong>Audit log</strong>
        <span>
          {integrity && (
            <span className={integrity.intact ? "good" : "err"} style={{ marginRight: 10, fontSize: 12 }}>
              {integrity.intact ? `Chain intact (${integrity.rows} entries)` : `TAMPERING DETECTED at entry #${integrity.first_bad_row_id}`}
            </span>
          )}
          <button className="ghost small" onClick={onVerify}>Verify integrity</button>{" "}
          <button className="ghost small" onClick={onRefresh}>Refresh</button>
        </span>
      </div>
      {entries.length === 0 ? (
        <div className="note">No decisions recorded yet.</div>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>#</th><th>Time (UTC)</th><th>Scan</th><th>Decision</th><th>Reviewer</th><th>Conf.</th><th>Vol. (mL)</th><th>Comment</th></tr>
            </thead>
            <tbody>
              {entries.map((e) => (
                <tr key={e.id}>
                  <td>{e.id}</td>
                  <td>{e.timestamp_utc.replace("T", " ").replace("+00:00", "")}</td>
                  <td>{e.filename || e.scan_id}</td>
                  <td className={e.decision === "approved" ? "good" : "err"}>{e.decision}</td>
                  <td>{e.reviewer}</td>
                  <td>{e.confidence != null ? `${Math.round(e.confidence * 100)}%` : "—"}</td>
                  <td>{e.tumour_volume_ml ?? "—"}</td>
                  <td>{e.comment || ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
