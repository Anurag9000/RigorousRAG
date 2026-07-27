"use strict";

const $ = (id) => document.getElementById(id);
const chatArea = $("chat-area");
const queryInput = $("query-input");
const sendBtn = $("send-btn");
const loader = $("loader");
const citationsArea = $("citations-area");
const fileInput = $("file-input");
const uploadList = $("upload-list");
const dropArea = $("drop-area");
const modelSelect = $("model-select");
const apiKeyInput = $("api-key-input");
const HISTORY_KEY = "rigorousrag_session_history_v4";
const API_KEY_KEY = "rigorousrag_session_api_key";
let appConfig = { auth_required: false, allowed_models: [], default_model: "", retain_uploads: false };
let allDocs = [];

function apiKey() {
  return sessionStorage.getItem(API_KEY_KEY) || "";
}

function apiHeaders(extra = {}) {
  const headers = { ...extra };
  const key = apiKey();
  if (key) headers["X-API-Key"] = key;
  return headers;
}

async function fetchApi(path, options = {}) {
  const request = { ...options, headers: apiHeaders(options.headers || {}) };
  const response = await fetch(path, request);
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
}

function clearNode(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function textElement(tag, text, className = "") {
  const element = document.createElement(tag);
  if (className) element.className = className;
  element.textContent = text == null ? "" : String(text);
  return element;
}

function safeExternalUrl(raw) {
  try {
    const url = new URL(String(raw || ""), window.location.origin);
    if (url.protocol === "http:" || url.protocol === "https:") return url.href;
  } catch { /* invalid URL */ }
  return null;
}

function appendInlineMarkdown(parent, rawText) {
  const text = String(rawText || "");
  const pattern = /(`[^`\n]+`|\*\*[^*\n]+\*\*|\[[^\]\n]+\]\(https?:\/\/[^)\s]+\))/g;
  let cursor = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > cursor) parent.appendChild(document.createTextNode(text.slice(cursor, match.index)));
    const token = match[0];
    if (token.startsWith("`")) {
      parent.appendChild(textElement("code", token.slice(1, -1)));
    } else if (token.startsWith("**")) {
      parent.appendChild(textElement("strong", token.slice(2, -2)));
    } else {
      const parts = /^\[([^\]]+)\]\((https?:\/\/[^)]+)\)$/.exec(token);
      const href = parts ? safeExternalUrl(parts[2]) : null;
      if (parts && href) {
        const link = textElement("a", parts[1]);
        link.href = href;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        parent.appendChild(link);
      } else {
        parent.appendChild(document.createTextNode(token));
      }
    }
    cursor = match.index + token.length;
  }
  if (cursor < text.length) parent.appendChild(document.createTextNode(text.slice(cursor)));
}

function isTableSeparator(line) {
  return /^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$/.test(line);
}

function tableCells(line) {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
}

function renderMarkdown(container, markdown) {
  clearNode(container);
  const lines = String(markdown || "").replace(/\r\n?/g, "\n").split("\n");
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    if (line.startsWith("```")) {
      const language = line.slice(3).trim();
      const codeLines = [];
      index += 1;
      while (index < lines.length && !lines[index].startsWith("```")) {
        codeLines.push(lines[index]);
        index += 1;
      }
      const pre = document.createElement("pre");
      const code = textElement("code", codeLines.join("\n"));
      if (language) code.dataset.language = language;
      pre.appendChild(code);
      container.appendChild(pre);
      index += 1;
      continue;
    }
    if (line.includes("|") && index + 1 < lines.length && isTableSeparator(lines[index + 1])) {
      const table = document.createElement("table");
      const head = document.createElement("thead");
      const headerRow = document.createElement("tr");
      for (const cell of tableCells(line)) {
        const th = document.createElement("th");
        appendInlineMarkdown(th, cell);
        headerRow.appendChild(th);
      }
      head.appendChild(headerRow);
      table.appendChild(head);
      const body = document.createElement("tbody");
      index += 2;
      while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
        const row = document.createElement("tr");
        for (const cell of tableCells(lines[index])) {
          const td = document.createElement("td");
          appendInlineMarkdown(td, cell);
          row.appendChild(td);
        }
        body.appendChild(row);
        index += 1;
      }
      table.appendChild(body);
      container.appendChild(table);
      continue;
    }
    const heading = /^(#{1,3})\s+(.+)$/.exec(line);
    if (heading) {
      const node = document.createElement(`h${heading[1].length}`);
      appendInlineMarkdown(node, heading[2]);
      container.appendChild(node);
      index += 1;
      continue;
    }
    if (/^\s*[-*]\s+/.test(line)) {
      const list = document.createElement("ul");
      while (index < lines.length && /^\s*[-*]\s+/.test(lines[index])) {
        const item = document.createElement("li");
        appendInlineMarkdown(item, lines[index].replace(/^\s*[-*]\s+/, ""));
        list.appendChild(item);
        index += 1;
      }
      container.appendChild(list);
      continue;
    }
    if (/^\s*\d+\.\s+/.test(line)) {
      const list = document.createElement("ol");
      while (index < lines.length && /^\s*\d+\.\s+/.test(lines[index])) {
        const item = document.createElement("li");
        appendInlineMarkdown(item, lines[index].replace(/^\s*\d+\.\s+/, ""));
        list.appendChild(item);
        index += 1;
      }
      container.appendChild(list);
      continue;
    }
    if (line.startsWith(">")) {
      const quote = document.createElement("blockquote");
      appendInlineMarkdown(quote, line.replace(/^>\s?/, ""));
      container.appendChild(quote);
      index += 1;
      continue;
    }
    if (!line.trim()) {
      index += 1;
      continue;
    }
    const paragraphLines = [line];
    index += 1;
    while (
      index < lines.length && lines[index].trim() &&
      !lines[index].startsWith("```") &&
      !/^(#{1,3})\s+/.test(lines[index]) &&
      !/^\s*[-*]\s+/.test(lines[index]) &&
      !/^\s*\d+\.\s+/.test(lines[index]) &&
      !lines[index].startsWith(">")
    ) {
      paragraphLines.push(lines[index]);
      index += 1;
    }
    const paragraph = document.createElement("p");
    appendInlineMarkdown(paragraph, paragraphLines.join("\n"));
    container.appendChild(paragraph);
  }
}

function appendMessage(text, isUser, warnings = [], persist = true) {
  const article = document.createElement("article");
  article.className = `message ${isUser ? "user" : "agent"}`;
  if (!isUser) article.appendChild(textElement("div", "RigorousRAG", "message-label"));
  const body = document.createElement("div");
  if (isUser) body.textContent = text;
  else renderMarkdown(body, text);
  article.appendChild(body);
  if (!isUser && Array.isArray(warnings) && warnings.length) {
    const warningBox = document.createElement("div");
    warningBox.className = "warnings";
    for (const warning of warnings) warningBox.appendChild(textElement("div", warning));
    article.appendChild(warningBox);
  }
  chatArea.appendChild(article);
  chatArea.scrollTop = chatArea.scrollHeight;
  if (persist) persistMessage(text, isUser, warnings);
}

function persistMessage(text, isUser, warnings = []) {
  try {
    const history = JSON.parse(sessionStorage.getItem(HISTORY_KEY) || "[]");
    history.push({ text: String(text), isUser: Boolean(isUser), warnings: warnings.slice(0, 10) });
    sessionStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(-100)));
  } catch { sessionStorage.removeItem(HISTORY_KEY); }
}

function restoreHistory() {
  try {
    const history = JSON.parse(sessionStorage.getItem(HISTORY_KEY) || "[]");
    if (!Array.isArray(history) || !history.length) return;
    clearNode(chatArea);
    for (const item of history) appendMessage(item.text || "", Boolean(item.isUser), item.warnings || [], false);
  } catch { sessionStorage.removeItem(HISTORY_KEY); }
}

function updateCitations(citations) {
  clearNode(citationsArea);
  if (!Array.isArray(citations) || !citations.length) {
    citationsArea.appendChild(textElement("div", "No cited evidence in this answer.", "empty"));
    return;
  }
  switchTab("left", "citations");
  for (const citation of citations) {
    const card = document.createElement("article");
    card.className = "citation-card";
    const head = document.createElement("div");
    head.className = "citation-head";
    head.appendChild(textElement("div", `${citation.label || ""} ${citation.title || "Untitled"}`, "citation-title"));
    head.appendChild(textElement("span", String(citation.source_type || "unknown").replaceAll("_", " "), "meta"));
    card.appendChild(head);
    const href = safeExternalUrl(citation.url);
    if (href) {
      const link = textElement("a", citation.url);
      link.href = href;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      card.appendChild(link);
    } else {
      card.appendChild(textElement("div", citation.url || "Local evidence", "meta"));
    }
    const location = [];
    if (citation.page_number) location.push(`page ${citation.page_number}`);
    if (citation.chunk_id) location.push(`chunk ${citation.chunk_id}`);
    if (location.length) card.appendChild(textElement("div", location.join(" · "), "meta"));
    const snippet = citation.quote || citation.snippet;
    if (snippet) card.appendChild(textElement("div", snippet, "citation-snippet"));
    citationsArea.appendChild(card);
  }
}

function setBusy(busy) {
  sendBtn.disabled = busy;
  loader.style.display = busy ? "block" : "none";
}

async function sendQuery() {
  const query = queryInput.value.trim();
  if (!query || sendBtn.disabled) return;
  appendMessage(query, true);
  queryInput.value = "";
  setBusy(true);
  try {
    const payload = await fetchApi("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, model: modelSelect.value || null }),
    });
    appendMessage(payload.answer || "No answer was returned.", false, payload.warnings || []);
    updateCitations(payload.citations || []);
  } catch (error) {
    appendMessage(`Request failed: ${error.message}`, false, error.status === 401 ? ["Enter a valid API key in the top bar."] : []);
  } finally {
    setBusy(false);
    queryInput.focus();
  }
}

function setQuery(text) {
  queryInput.value = text;
  queryInput.focus();
}

async function pollJob(jobId, item, filename) {
  for (let attempt = 0; attempt < 120; attempt += 1) {
    await new Promise((resolve) => setTimeout(resolve, 2500));
    try {
      const status = await fetchApi(`/status/${encodeURIComponent(jobId)}`);
      item.lastChild.textContent = status.status === "processing" ? "Processing" : status.status;
      if (status.status === "success") {
        item.lastChild.style.color = "var(--success)";
        if (status.doc_id) item.title = `Document ID: ${status.doc_id}`;
        await loadDocList();
        return;
      }
      if (status.status === "failed") {
        item.lastChild.style.color = "var(--danger)";
        item.title = status.message || "Ingestion failed";
        return;
      }
    } catch (error) {
      if (error.status === 401 || error.status === 404) {
        item.lastChild.textContent = "Status unavailable";
        item.lastChild.style.color = "var(--danger)";
        return;
      }
    }
  }
  item.lastChild.textContent = "Still processing; refresh Docs later";
  item.lastChild.style.color = "var(--warning)";
}

async function handleFiles(files) {
  const values = Array.from(files || []);
  if (!values.length) return;
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
      const response = await fetchApi("/ingest", { method: "POST", body: form });
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
}

function renderDocList(documents) {
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
    const name = textElement("div", doc.filename || doc.doc_id, "doc-title");
    head.appendChild(name);
    const deleteButton = textElement("button", "Delete", "btn small danger");
    deleteButton.type = "button";
    deleteButton.addEventListener("click", async (event) => {
      event.stopPropagation();
      if (!window.confirm(`Delete ${doc.filename || doc.doc_id}?`)) return;
      try {
        await fetchApi(`/docs/${encodeURIComponent(doc.doc_id)}`, { method: "DELETE" });
        await loadDocList();
      } catch (error) { window.alert(error.message); }
    });
    head.appendChild(deleteButton);
    card.appendChild(head);
    card.appendChild(textElement("div", `ID: ${doc.doc_id}`, "meta"));
    if (doc.mime_type) card.appendChild(textElement("div", doc.mime_type, "meta"));
    if (doc.llm_summary) card.appendChild(textElement("div", doc.llm_summary, "summary"));
    card.addEventListener("click", () => setQuery(`Search uploaded document ID ${doc.doc_id} for: `));
    list.appendChild(card);
  }
}

async function loadDocList() {
  const list = $("doc-list");
  clearNode(list);
  list.appendChild(textElement("div", "Loading documents…", "empty"));
  try {
    allDocs = await fetchApi("/docs/list");
    filterDocs();
  } catch (error) {
    clearNode(list);
    list.appendChild(textElement("div", `Could not load documents: ${error.message}`, "empty"));
  }
}

function filterDocs() {
  const query = $("doc-search").value.trim().toLowerCase();
  renderDocList(allDocs.filter((doc) => {
    const haystack = `${doc.filename || ""} ${doc.llm_summary || ""} ${doc.doc_id || ""}`.toLowerCase();
    return haystack.includes(query);
  }));
}

function switchTab(group, name) {
  document.querySelectorAll(`[data-tab-group="${group}"]`).forEach((button) => {
    button.setAttribute("aria-selected", button.dataset.tab === name ? "true" : "false");
  });
  const prefix = group === "left" ? "tab-" : "tool-";
  document.querySelectorAll(group === "left" ? ".left-panel .tab-pane" : ".right-panel .tool-pane").forEach((pane) => pane.classList.remove("active"));
  const target = $(`${prefix}${name}`);
  if (target) target.classList.add("active");
  if (group === "left" && name === "docs") loadDocList();
}

function togglePanel(panelId, open) {
  const panel = $(panelId);
  panel.classList.toggle("open", open);
  const anyOpen = $("left-panel").classList.contains("open") || $("right-panel").classList.contains("open");
  $("sidebar-overlay").classList.toggle("open", anyOpen);
}

function showToolResult(id, value) {
  const element = $(id);
  element.hidden = false;
  element.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

async function runEntailment() {
  const payload = {
    claim_text: $("ve-claim").value.trim(),
    figure_id: $("ve-figure").value.trim(),
    doc_id: $("ve-docid").value.trim(),
  };
  if (!payload.claim_text || !payload.figure_id || !payload.doc_id) return window.alert("Claim, figure label, and document ID are required.");
  try {
    const result = await fetchApi("/tool/visual-entailment", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    showToolResult("ve-result", result);
  } catch (error) { showToolResult("ve-result", `Error: ${error.message}`); }
}

async function runProtocol() {
  const payload = { text: $("proto-text").value.trim(), doc_id: $("proto-docid").value.trim() };
  if (!payload.text) return window.alert("Methods text is required.");
  try {
    const result = await fetchApi("/tool/protocol", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    showToolResult("proto-result", result);
  } catch (error) { showToolResult("proto-result", `Error: ${error.message}`); }
}

async function runBibtex() {
  const payload = {
    title: $("bib-title").value.trim(), authors: $("bib-authors").value.trim(),
    year: Number.parseInt($("bib-year").value, 10) || null,
    journal: $("bib-journal").value.trim(), doi: $("bib-doi").value.trim(), entry_type: "article",
  };
  if (!payload.title) return window.alert("Title is required.");
  try {
    const result = await fetchApi("/tool/bibtex", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    showToolResult("bib-result", result.bibtex || "");
  } catch (error) { showToolResult("bib-result", `Error: ${error.message}`); }
}

async function loadConfig() {
  try {
    appConfig = await fetchApi("/config");
    clearNode(modelSelect);
    for (const model of appConfig.allowed_models || []) {
      const option = textElement("option", model);
      option.value = model;
      option.selected = model === appConfig.default_model;
      modelSelect.appendChild(option);
    }
    $("auth-box").style.display = appConfig.auth_required ? "flex" : "none";
    $("retention-note").textContent = appConfig.retain_uploads
      ? "Original uploads are retained for owner-scoped figure tools. Best-effort PII masking is not guaranteed anonymization."
      : "Original uploads are deleted after indexing. Best-effort PII masking is not guaranteed anonymization.";
  } catch { /* defaults remain usable */ }
}

async function checkHealth() {
  try {
    await fetchApi("/health");
    $("server-status-dot").classList.add("online");
    $("server-status-text").textContent = "Online";
  } catch {
    $("server-status-dot").classList.remove("online");
    $("server-status-text").textContent = "Offline";
  }
}

function bindEvents() {
  document.querySelectorAll("[data-tab-group]").forEach((button) => button.addEventListener("click", () => switchTab(button.dataset.tabGroup, button.dataset.tab)));
  document.querySelectorAll("[data-prompt]").forEach((button) => button.addEventListener("click", () => setQuery(button.dataset.prompt)));
  $("send-btn").addEventListener("click", sendQuery);
  queryInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendQuery(); }
  });
  $("clear-history").addEventListener("click", () => { sessionStorage.removeItem(HISTORY_KEY); clearNode(chatArea); appendMessage("Conversation cleared.", false, [], false); });
  $("save-api-key").addEventListener("click", async () => {
    const key = apiKeyInput.value.trim();
    if (key) sessionStorage.setItem(API_KEY_KEY, key); else sessionStorage.removeItem(API_KEY_KEY);
    apiKeyInput.value = "";
    await checkHealth();
    await loadDocList();
  });
  $("choose-files-btn").addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => handleFiles(fileInput.files));
  ["dragenter", "dragover", "dragleave", "drop"].forEach((name) => dropArea.addEventListener(name, (event) => { event.preventDefault(); event.stopPropagation(); }));
  dropArea.addEventListener("dragover", () => dropArea.classList.add("dragover"));
  dropArea.addEventListener("dragleave", () => dropArea.classList.remove("dragover"));
  dropArea.addEventListener("drop", (event) => { dropArea.classList.remove("dragover"); handleFiles(event.dataTransfer.files); });
  dropArea.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") fileInput.click(); });
  $("doc-search").addEventListener("input", filterDocs);
  $("open-sidebar").addEventListener("click", () => togglePanel("left-panel", true));
  $("open-tools").addEventListener("click", () => togglePanel("right-panel", true));
  $("close-tools").addEventListener("click", () => togglePanel("right-panel", false));
  $("sidebar-overlay").addEventListener("click", () => { togglePanel("left-panel", false); togglePanel("right-panel", false); });
  $("run-entailment").addEventListener("click", runEntailment);
  $("run-protocol").addEventListener("click", runProtocol);
  $("run-debate").addEventListener("click", () => {
    const claim = $("debate-claim").value.trim();
    const evidence = $("debate-evidence").value.trim();
    if (!claim || !evidence) return window.alert("A claim and evidence context are required.");
    setQuery(`Run an evidence-grounded scientific debate on this claim: ${claim}\n\nOriginal evidence:\n${evidence}`);
    sendQuery();
  });
  $("run-bibtex").addEventListener("click", runBibtex);
  $("copy-bibtex").addEventListener("click", async () => {
    const text = $("bib-result").textContent || "";
    if (text) await navigator.clipboard.writeText(text);
  });
}

async function boot() {
  bindEvents();
  restoreHistory();
  await loadConfig();
  await checkHealth();
  setInterval(checkHealth, 30_000);
}

boot();
