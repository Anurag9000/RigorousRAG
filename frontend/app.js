// ============================================================
// RigorousRAG — app.js
// Full frontend integration: chat, uploads, tool panels,
// doc list, server health, model selector, session persistence.
// ============================================================

// ===== DOM References =====
const chatArea      = document.getElementById("chat-area");
const queryInput    = document.getElementById("query-input");
const sendBtn       = document.getElementById("send-btn");
const loader        = document.getElementById("loader");
const citationsArea = document.getElementById("citations-area");
const fileInput     = document.getElementById("file-input");
const uploadList    = document.getElementById("upload-list");
const dropArea      = document.getElementById("drop-area");
const modelSelect   = document.getElementById("model-select");

// ===== Markdown =====
marked.setOptions({ breaks: true, gfm: true });

// ============================================================
// Session Persistence (localStorage)
// ============================================================
const STORAGE_KEY = "rigorousrag_v2_history";

function restoreSession() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return;
        const history = JSON.parse(raw);
        history.forEach(({ text, isUser }) => appendMessage(text, isUser, false));
    } catch { localStorage.removeItem(STORAGE_KEY); }
}

function persistMessage(text, isUser) {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        const history = raw ? JSON.parse(raw) : [];
        history.push({ text, isUser });
        if (history.length > 200) history.splice(0, history.length - 200);
        localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    } catch { /* quota exceeded — silently degrade */ }
}

function clearHistory() {
    localStorage.removeItem(STORAGE_KEY);
    chatArea.innerHTML = "";
    appendMessage("**History cleared.** Ask a new research question.", false, false);
    citationsArea.innerHTML = `<div class="empty-state">Citations will appear after the next query.</div>`;
}

// ============================================================
// Server Health Indicator
// ============================================================
const statusDot  = document.getElementById("server-status-dot");
const statusText = document.getElementById("server-status-text");

async function checkServerHealth() {
    try {
        // /docs/list returns quickly even with an empty DB
        const r = await fetch("/docs/list", { method: "GET", signal: AbortSignal.timeout(4000) });
        if (r.ok || r.status === 200) {
            statusDot.classList.remove("offline");
            statusText.textContent = "Online";
        } else {
            throw new Error(r.status);
        }
    } catch {
        statusDot.classList.add("offline");
        statusText.textContent = "Offline";
    }
}
checkServerHealth();
setInterval(checkServerHealth, 30_000);   // re-check every 30 s

// ============================================================
// Chat Helpers
// ============================================================

function setQuery(text) {
    queryInput.value = text;
    queryInput.focus();
    queryInput.style.height = "auto";
    queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + "px";
}

function handleKey(e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendQuery(); }
}

queryInput.addEventListener("input", () => {
    queryInput.style.height = "auto";
    queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + "px";
});

function appendMessage(text, isUser, persist = true) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${isUser ? "user-msg" : "agent-msg"}`;
    if (!isUser) {
        const lbl = document.createElement("div");
        lbl.className = "msg-label";
        lbl.textContent = "RigorousRAG";
        msgDiv.appendChild(lbl);
    }
    const body = document.createElement("div");
    body.innerHTML = isUser ? escapeHTML(text) : marked.parse(text);
    msgDiv.appendChild(body);
    chatArea.appendChild(msgDiv);
    chatArea.scrollTop = chatArea.scrollHeight;
    if (persist) persistMessage(text, isUser);
}

function escapeHTML(str) {
    return str.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function showTyping() {
    const el = document.createElement("div");
    el.className = "message agent-msg";
    el.id = "typing-indicator";
    el.innerHTML = `<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>`;
    chatArea.appendChild(el);
    chatArea.scrollTop = chatArea.scrollHeight;
}

function removeTyping() {
    document.getElementById("typing-indicator")?.remove();
}

// ============================================================
// updateCitations — switches to Sources tab automatically
// ============================================================
function updateCitations(citations) {
    citationsArea.innerHTML = "";
    if (!citations || citations.length === 0) {
        citationsArea.innerHTML = `<div class="empty-state">No citations for this response.</div>`;
        return;
    }
    // Auto-switch to citations tab
    switchLeftTab("citations", document.querySelectorAll(".left-tab")[2]);

    citations.forEach(c => {
        const card = document.createElement("div");
        card.className = "citation-card";
        card.setAttribute("data-source", c.source_type);

        const displayUrl = c.url.startsWith("local://") ? "Internal Index" : c.url;
        card.innerHTML = `
            <div class="citation-header">
                <span class="citation-title">${c.label} ${escapeHTML(c.title.substring(0, 40))}${c.title.length > 40 ? "…" : ""}</span>
                <span class="source-tag">${c.source_type.replace(/_/g, " ")}</span>
            </div>
            <div class="citation-url">
                <a href="${c.url.startsWith("http") ? c.url : "#"}" target="_blank" rel="noopener">${escapeHTML(displayUrl.substring(0, 60))}</a>
            </div>
            <div class="citation-snippet">"${escapeHTML(c.snippet.substring(0, 200))}${c.snippet.length > 200 ? "…" : ""}"</div>
        `;
        citationsArea.appendChild(card);
    });
}

// ============================================================
// Main Query Submission
// ============================================================
async function sendQuery() {
    const query = queryInput.value.trim();
    if (!query || sendBtn.disabled) return;

    appendMessage(query, true);
    queryInput.value = "";
    queryInput.style.height = "2.8rem";
    sendBtn.disabled = true;
    loader.style.display = "block";
    showTyping();

    try {
        const model = modelSelect.value;
        const res = await fetch("/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query, model }),
        });
        const data = await res.json();
        removeTyping();

        if (res.ok) {
            appendMessage(data.answer, false);
            updateCitations(data.citations);
        } else {
            appendMessage(`**Backend Error (${res.status}):**\n\`\`\`\n${data.detail || "Unknown error"}\n\`\`\``, false);
        }
    } catch (err) {
        removeTyping();
        appendMessage(`**Network Error:** Could not reach the backend.\nMake sure the server is running (\`python server.py\`).`, false);
    } finally {
        sendBtn.disabled = false;
        loader.style.display = "none";
        queryInput.focus();
    }
}

// ============================================================
// Visual Entailment shortcut (from action chip)
// ============================================================
function openVisualEntailment() {
    switchToolTab("entailment", document.querySelectorAll(".tool-tab")[0]);
    // Scroll right panel into view on mobile
    document.querySelector(".right-panel")?.scrollIntoView({ behavior: "smooth" });
    document.getElementById("ve-claim").focus();
}

// ============================================================
// Ingestion with Job-Status Polling
// ============================================================
async function pollJobStatus(jobId, listItem, filename) {
    const MAX = 40;   // 40 × 3 s = 2 min
    for (let i = 0; i < MAX; i++) {
        await new Promise(r => setTimeout(r, 3000));
        try {
            const res = await fetch(`/status/${jobId}`);
            if (!res.ok) continue;
            const data = await res.json();
            if (data.status === "success") {
                setUploadItemStatus(listItem, filename, "Indexed ✓", "var(--success)");
                // Refresh doc list if on docs tab
                if (document.getElementById("tab-docs").classList.contains("active")) loadDocList();
                return;
            }
            if (data.status === "failed") {
                setUploadItemStatus(listItem, filename, `Failed: ${data.message || ""}`, "var(--danger)");
                return;
            }
        } catch { /* network hiccup — keep polling */ }
    }
    setUploadItemStatus(listItem, filename, "Timed out", "var(--warning)");
}

function setUploadItemStatus(listItem, filename, statusText, color) {
    listItem.innerHTML = `
        <span class="upload-item-name" title="${escapeHTML(filename)}">${escapeHTML(filename.substring(0, 28))}${filename.length > 28 ? "…" : ""}</span>
        <span class="upload-item-status" style="color:${color};">${statusText}</span>
    `;
}

async function handleFiles(files) {
    if (!files.length) return;
    // Switch to upload tab
    switchLeftTab("upload", document.querySelectorAll(".left-tab")[0]);

    for (const file of Array.from(files)) {
        const listItem = document.createElement("div");
        listItem.className = "upload-item";
        setUploadItemStatus(listItem, file.name, "Uploading…", "var(--warning)");
        uploadList.insertBefore(listItem, uploadList.firstChild);

        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("/ingest", { method: "POST", body: formData });
            const data = await res.json();
            if (res.ok && data.job_id) {
                setUploadItemStatus(listItem, file.name, "Ingesting…", "var(--warning)");
                pollJobStatus(data.job_id, listItem, file.name);
            } else {
                setUploadItemStatus(listItem, file.name, "Failed", "var(--danger)");
            }
        } catch {
            setUploadItemStatus(listItem, file.name, "Network Error", "var(--danger)");
        }
    }
}

// ============================================================
// Document Library (Docs Tab)
// ============================================================
let _allDocs = [];

async function loadDocList() {
    const docList = document.getElementById("doc-list");
    docList.innerHTML = `<div class="empty-state">Loading…</div>`;
    try {
        const res = await fetch("/docs/list");
        if (!res.ok) throw new Error(res.status);
        _allDocs = await res.json();
        renderDocList(_allDocs);
    } catch (err) {
        docList.innerHTML = `<div class="empty-state">Could not load documents.<br><small>${err}</small></div>`;
    }
}

function renderDocList(docs) {
    const docList = document.getElementById("doc-list");
    if (!docs.length) {
        docList.innerHTML = `<div class="empty-state">No documents indexed yet.<br>Upload files in the Upload tab.</div>`;
        return;
    }
    docList.innerHTML = "";
    docs.forEach(doc => {
        const card = document.createElement("div");
        card.className = "doc-card";
        card.title = `Click to pre-fill doc ID: ${doc.doc_id}`;
        const ext = doc.mime_type ? doc.mime_type.split("/").pop().toUpperCase() : "DOC";
        const summary = doc.llm_summary || "(No summary available)";
        card.innerHTML = `
            <div class="doc-card-name">📄 ${escapeHTML(doc.filename)}</div>
            <div class="doc-card-meta">
                <span class="badge badge-success">${ext}</span> &nbsp;
                <span class="doc-id-badge" title="Click to copy" onclick="copyDocId('${doc.doc_id}', event)">${doc.doc_id}</span>
            </div>
            <div class="doc-card-summary">${escapeHTML(summary)}</div>
        `;
        // Click card → pre-fill query with this doc
        card.addEventListener("click", () => {
            setQuery(`Search my uploaded documents for (doc: ${doc.doc_id}): `);
        });
        docList.appendChild(card);
    });
}

function filterDocs() {
    const q = document.getElementById("doc-search").value.toLowerCase();
    renderDocList(_allDocs.filter(d =>
        d.filename.toLowerCase().includes(q) || (d.llm_summary || "").toLowerCase().includes(q)
    ));
}

function copyDocId(id, e) {
    e.stopPropagation();
    navigator.clipboard.writeText(id).then(() => {
        const badge = e.target;
        const orig = badge.textContent;
        badge.textContent = "Copied!";
        setTimeout(() => { badge.textContent = orig; }, 1200);
    });
}

// ============================================================
// Left Panel Tabs
// ============================================================
function switchLeftTab(tab, el) {
    document.querySelectorAll(".left-tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".left-tab-pane").forEach(p => p.classList.remove("active"));
    el?.classList.add("active");
    document.getElementById(`tab-${tab}`)?.classList.add("active");

    if (tab === "docs") loadDocList();
}

// ============================================================
// Right Panel Tool Tabs
// ============================================================
function switchToolTab(tab, el) {
    document.querySelectorAll(".tool-tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tool-pane").forEach(p => p.classList.remove("active"));
    el?.classList.add("active");
    document.getElementById(`tool-${tab}`)?.classList.add("active");
}

// ============================================================
// Tool: Visual Entailment
// ============================================================
async function runVisualEntailment() {
    const claim   = document.getElementById("ve-claim").value.trim();
    const figure  = document.getElementById("ve-figure").value.trim();
    const docId   = document.getElementById("ve-docid").value.trim();

    if (!claim || !figure || !docId) {
        alert("Please fill in Claim, Figure ID, and Document ID.");
        return;
    }

    const resultSection = document.getElementById("ve-result-section");
    const resultEl      = document.getElementById("ve-result");
    resultSection.style.display = "none";
    resultEl.textContent = "Running…";

    try {
        const res  = await fetch("/tool/visual-entailment", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ claim_text: claim, figure_id: figure, doc_id: docId }),
        });
        const data = await res.json();
        resultSection.style.display = "block";

        if (data.error) {
            resultEl.innerHTML = `<span style="color:var(--danger)"><strong>Error:</strong> ${escapeHTML(data.error)}</span>`;
            return;
        }

        const verdictColors = {
            supports: "var(--success)", contradicts: "var(--danger)",
            insufficient: "var(--warning)", uncertain: "#94a3b8",
        };
        const color = verdictColors[data.verdict] || "#94a3b8";
        resultEl.innerHTML = `<strong style="color:${color}">▶ ${(data.verdict || "?").toUpperCase()}</strong>\n\nConfidence: ${((data.confidence || 0) * 100).toFixed(0)}%\n\n${data.rationale || ""}`;

        // Also post to chat for record
        appendMessage(
            `**Visual Entailment Result**\n- **Claim:** ${claim}\n- **Figure:** ${figure} in \`${docId}\`\n- **Verdict:** ${(data.verdict || "?").toUpperCase()} (${((data.confidence || 0) * 100).toFixed(0)}% confidence)\n- **Rationale:** ${data.rationale || "N/A"}`,
            false
        );
    } catch (err) {
        resultSection.style.display = "block";
        resultEl.textContent = `Network error: ${err}`;
    }
}

// ============================================================
// Tool: Protocol Extraction
// ============================================================
async function runProtocolExtraction() {
    const text  = document.getElementById("proto-text").value.trim();
    const docId = document.getElementById("proto-docid").value.trim();

    if (!text) { alert("Please paste the methods section text."); return; }

    const resultSection = document.getElementById("proto-result-section");
    const resultEl      = document.getElementById("proto-result");
    resultSection.style.display = "none";
    resultEl.textContent = "Extracting…";

    try {
        const res  = await fetch("/tool/protocol", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, doc_id: docId }),
        });
        const data = await res.json();
        resultSection.style.display = "block";

        if (data.error) {
            resultEl.innerHTML = `<span style="color:var(--danger)"><strong>Error:</strong> ${escapeHTML(data.error)}</span>`;
            return;
        }
        if (!data.steps) {
            resultEl.textContent = JSON.stringify(data, null, 2);
            return;
        }

        resultEl.textContent = data.steps.map((s, i) => {
            let line = `Step ${i + 1}: ${s.description}`;
            if (s.temperature) line += ` | Temp: ${s.temperature}`;
            if (s.time)        line += ` | Time: ${s.time}`;
            if (s.reagent)     line += ` | Reagent: ${s.reagent}`;
            if (s.notes)       line += `\n  Notes: ${s.notes}`;
            return line;
        }).join("\n\n");

        // Mirror to chat
        const stepsText = data.steps.map((s, i) =>
            `${i + 1}. **${s.description}**${s.temperature ? ` *(${s.temperature})* ` : ""}${s.time ? ` — ${s.time}` : ""}`
        ).join("\n");
        appendMessage(`**Protocol Extracted** (${data.steps.length} steps, method: \`${data.metadata?.extraction_method || "?"}\`)\n\n${stepsText}`, false);
    } catch (err) {
        resultSection.style.display = "block";
        resultEl.textContent = `Network error: ${err}`;
    }
}

// ============================================================
// Tool: Scientific Debate (via main agent)
// ============================================================
async function runDebate() {
    const claim    = document.getElementById("debate-claim").value.trim();
    const evidence = document.getElementById("debate-evidence").value.trim();

    if (!claim) { alert("Enter a claim to debate."); return; }

    const prompt = evidence
        ? `Run a rigorous scientific debate on the claim: "${claim}"\n\nContext/Evidence:\n${evidence}`
        : `Run a rigorous scientific debate on the claim: "${claim}"`;

    setQuery(prompt);
    await sendQuery();
}

// ============================================================
// Tool: BibTeX Export
// ============================================================
async function runBibTeX() {
    const title   = document.getElementById("bib-title").value.trim();
    const authors = document.getElementById("bib-authors").value.trim();
    const year    = parseInt(document.getElementById("bib-year").value) || null;
    const doi     = document.getElementById("bib-doi").value.trim();
    const journal = document.getElementById("bib-journal").value.trim();

    if (!title) { alert("Title is required."); return; }

    const resultSection = document.getElementById("bib-result-section");
    const resultEl      = document.getElementById("bib-result");
    resultSection.style.display = "none";
    resultEl.textContent = "Generating…";

    try {
        const res = await fetch("/tool/bibtex", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title, authors, year, doi, journal }),
        });
        const data = await res.json();
        resultSection.style.display = "block";
        if (data.error) {
            resultEl.innerHTML = `<span style="color:var(--danger)"><strong>Error:</strong> ${escapeHTML(data.error)}</span>`;
            return;
        }
        resultEl.textContent = data.bibtex || JSON.stringify(data, null, 2);
    } catch (err) {
        resultSection.style.display = "block";
        resultEl.textContent = `Network error: ${err}`;
    }
}

function copyBibTeX() {
    const text = document.getElementById("bib-result").textContent;
    navigator.clipboard.writeText(text).then(() => {
        const btn = document.querySelector("#tool-bibtex .tool-section-title button");
        if (btn) { const orig = btn.textContent; btn.textContent = "Copied!"; setTimeout(() => btn.textContent = orig, 1200); }
    });
}

// ============================================================
// File inputs & drag-and-drop
// ============================================================
fileInput.addEventListener("change", e => handleFiles(e.target.files));

["dragenter", "dragover", "dragleave", "drop"].forEach(ev =>
    dropArea.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); }, false)
);
dropArea.addEventListener("dragover",  () => dropArea.classList.add("dragover"));
dropArea.addEventListener("dragleave", () => dropArea.classList.remove("dragover"));
dropArea.addEventListener("drop", e => { dropArea.classList.remove("dragover"); handleFiles(e.dataTransfer.files); });

// ============================================================
// Sidebar Toggle (mobile)
// ============================================================
function toggleSidebar() {
    const panel = document.getElementById("left-panel");
    const overlay = document.getElementById("sidebar-overlay");
    panel.classList.toggle("open");
    overlay.style.display = panel.classList.contains("open") ? "block" : "none";
}

document.getElementById("sidebar-overlay").addEventListener("click", () => {
    document.getElementById("left-panel").classList.remove("open");
    document.getElementById("sidebar-overlay").style.display = "none";
});

// ============================================================
// Boot
// ============================================================
restoreSession();