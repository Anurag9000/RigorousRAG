"use strict";

const JOB_STATUS_LABELS = {
  queued: "Queued",
  processing: "Processing",
  finalizing: "Finalizing",
  success: "Indexed",
  failed: "Failed",
};

const JOB_STATUS_COLORS = {
  queued: "var(--warning)",
  processing: "var(--warning)",
  finalizing: "var(--warning)",
  success: "var(--success)",
  failed: "var(--danger)",
};

const DEFAULT_CLIENT_REQUEST_TIMEOUT_MS = 135_000;
const MAX_CLIENT_REQUEST_TIMEOUT_MS = 600_000;
const MAX_CLIENT_UPLOAD_FILES = 100;

function boundedClientTimeout(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return DEFAULT_CLIENT_REQUEST_TIMEOUT_MS;
  return Math.max(1000, Math.min(parsed, MAX_CLIENT_REQUEST_TIMEOUT_MS));
}

fetchApi = async function fetchApiWithDeadline(path, options = {}) {
  const source = options && typeof options === "object" ? options : {};
  const timeoutMs = boundedClientTimeout(source.timeoutMs);
  const requestOptions = { ...source };
  delete requestOptions.timeoutMs;

  const controller = new AbortController();
  const externalSignal = requestOptions.signal;
  let externalAbort = null;
  if (externalSignal && typeof externalSignal.addEventListener === "function") {
    if (externalSignal.aborted) controller.abort();
    else {
      externalAbort = () => controller.abort();
      externalSignal.addEventListener("abort", externalAbort, { once: true });
    }
  }
  requestOptions.signal = controller.signal;
  requestOptions.headers = apiHeaders(requestOptions.headers || {});
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(path, requestOptions);
    let payload = null;
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      try { payload = await response.json(); } catch { payload = null; }
    } else {
      payload = await response.text();
    }
    if (!response.ok) {
      const message = payload && typeof payload === "object" ? payload.detail : payload;
      const error = new Error(message || `Request failed with status ${response.status}.`);
      error.status = response.status;
      throw error;
    }
    return payload;
  } catch (error) {
    if (controller.signal.aborted && !(externalSignal && externalSignal.aborted)) {
      const timeoutError = new Error("Request timed out before the server responded.");
      timeoutError.status = 408;
      throw timeoutError;
    }
    throw error;
  } finally {
    window.clearTimeout(timer);
    if (externalAbort && externalSignal) {
      externalSignal.removeEventListener("abort", externalAbort);
    }
  }
};

function boundedUploadFiles(files) {
  const selected = [];
  if (!files || typeof files[Symbol.iterator] !== "function") return selected;
  for (const file of files) {
    if (selected.length >= MAX_CLIENT_UPLOAD_FILES) break;
    if (file && typeof file.name === "string") selected.push(file);
  }
  return selected;
}

handleFiles = async function handleBoundedFiles(files) {
  const values = boundedUploadFiles(files);
  if (!values.length) return;
  const reportedLength = Number(files && files.length);
  if (Number.isFinite(reportedLength) && reportedLength > MAX_CLIENT_UPLOAD_FILES) {
    window.alert(`Only the first ${MAX_CLIENT_UPLOAD_FILES} files will be uploaded.`);
  }
  switchTab("left", "upload");
  for (const file of values) {
    const item = document.createElement("div");
    item.className = "upload-item";
    item.appendChild(textElement("span", file.name));
    const status = textElement("span", "Uploading", "meta");
    item.appendChild(status);
    uploadList.prepend(item);
    const form = new FormData();
    form.append("file", file, file.name);
    try {
      const response = await fetchApi("/ingest", {
        method: "POST",
        body: form,
        timeoutMs: MAX_CLIENT_REQUEST_TIMEOUT_MS,
      });
      status.textContent = "Processing";
      status.style.color = "var(--warning)";
      pollJob(response.job_id, item, file.name);
    } catch (error) {
      status.textContent = "Failed";
      status.style.color = "var(--danger)";
      item.title = error.message;
    }
  }
  fileInput.value = "";
};

pollJob = async function pollDurableJob(jobId, item) {
  for (let attempt = 0; attempt < 240; attempt += 1) {
    await new Promise((resolve) => setTimeout(resolve, 2500));
    try {
      const status = await fetchApi(`/status/${encodeURIComponent(jobId)}`, {
        timeoutMs: 15_000,
      });
      const state = String(status.status || "unknown").toLowerCase();
      item.lastChild.textContent = JOB_STATUS_LABELS[state] || state;
      item.lastChild.style.color = JOB_STATUS_COLORS[state] || "var(--warning)";
      if (status.message) item.title = status.message;
      if (state === "success") {
        if (status.doc_id) item.title = `Document ID: ${status.doc_id}`;
        await loadDocList();
        return;
      }
      if (state === "failed") return;
    } catch (error) {
      if (error.status === 401 || error.status === 404) {
        item.lastChild.textContent = "Status unavailable";
        item.lastChild.style.color = "var(--danger)";
        return;
      }
    }
  }
  item.lastChild.textContent = "Still queued or processing; refresh Docs later";
  item.lastChild.style.color = "var(--warning)";
};

function openFigureToolForDocument(doc) {
  if (!doc.visual_source_available) {
    window.alert(
      doc.source_retained
        ? "This retained source is not an eligible PDF for figure checks."
        : "This document has text evidence only. Re-ingest it while source retention is enabled to use figure checks.",
    );
    return;
  }
  $("ve-docid").value = doc.doc_id || "";
  switchTab("tools", "entailment");
  togglePanel("right-panel", true);
  $("ve-figure").focus();
}

renderDocList = function renderLifecycleAwareDocuments(documents) {
  const list = $("doc-list");
  clearNode(list);
  if (!Array.isArray(documents) || !documents.length) {
    list.appendChild(textElement("div", "No indexed documents for this account.", "empty"));
    return;
  }
  for (const doc of documents.slice(0, 5000)) {
    if (!doc || typeof doc !== "object") continue;
    const card = document.createElement("article");
    card.className = "doc-card";
    const head = document.createElement("div");
    head.className = "doc-card-head";
    head.appendChild(textElement("div", doc.filename || doc.doc_id, "doc-title"));

    const actions = document.createElement("div");
    actions.className = "doc-actions";
    const visualEligible = Boolean(doc.visual_source_available);
    const sourceRetained = Boolean(doc.source_retained);
    const figureButton = textElement(
      "button",
      visualEligible ? "Figure tool" : (sourceRetained ? "Visual unavailable" : "No visual source"),
      "btn small",
    );
    figureButton.type = "button";
    figureButton.disabled = !visualEligible;
    figureButton.title = visualEligible
      ? "Open this retained PDF; identity and complexity are verified when the tool runs"
      : (sourceRetained ? "The retained source is not currently eligible for PDF figure analysis" : "The original source was not retained");
    figureButton.addEventListener("click", (event) => {
      event.stopPropagation();
      openFigureToolForDocument(doc);
    });
    actions.appendChild(figureButton);

    const deleteButton = textElement("button", "Delete", "btn small danger");
    deleteButton.type = "button";
    deleteButton.addEventListener("click", async (event) => {
      event.stopPropagation();
      if (!window.confirm(`Delete ${doc.filename || doc.doc_id}?`)) return;
      try {
        await fetchApi(`/docs/${encodeURIComponent(doc.doc_id)}`, {
          method: "DELETE",
          timeoutMs: 30_000,
        });
        await loadDocList();
      } catch (error) {
        window.alert(error.message);
      }
    });
    actions.appendChild(deleteButton);
    head.appendChild(actions);
    card.appendChild(head);

    card.appendChild(textElement("div", `ID: ${doc.doc_id}`, "meta"));
    if (doc.mime_type) card.appendChild(textElement("div", doc.mime_type, "meta"));
    card.appendChild(
      textElement(
        "div",
        visualEligible
          ? "Visual PDF eligible; identity and limits verified on use"
          : (sourceRetained ? "Source retained; visual analysis unavailable" : "Text evidence only"),
        "meta",
      ),
    );
    if (doc.llm_summary) card.appendChild(textElement("div", doc.llm_summary, "summary"));
    card.addEventListener("click", () => {
      setQuery(`Search uploaded document ID ${doc.doc_id} for: `);
    });
    list.appendChild(card);
  }
};
