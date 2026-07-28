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

pollJob = async function pollDurableJob(jobId, item) {
  for (let attempt = 0; attempt < 240; attempt += 1) {
    await new Promise((resolve) => setTimeout(resolve, 2500));
    try {
      const status = await fetchApi(`/status/${encodeURIComponent(jobId)}`);
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
  if (!documents.length) {
    list.appendChild(textElement("div", "No indexed documents for this account.", "empty"));
    return;
  }
  for (const doc of documents) {
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
        await fetchApi(`/docs/${encodeURIComponent(doc.doc_id)}`, { method: "DELETE" });
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
