// ===== DOM references =====
const chatArea       = document.getElementById("chat-area");
const queryInput     = document.getElementById("query-input");
const sendBtn        = document.getElementById("send-btn");
const loader         = document.getElementById("loader");
const citationsArea  = document.getElementById("citations-area");
const fileInput      = document.getElementById("file-input");
const uploadList     = document.getElementById("upload-list");
const dropArea       = document.getElementById("drop-area");
const sidebarToggle  = document.getElementById("sidebar-toggle");
const sidebar        = document.querySelector(".sidebar");

// ===== Markdown rendering =====
marked.setOptions({ breaks: true, gfm: true });

// ===== Session persistence (localStorage) =====
const STORAGE_KEY = "rigorousrag_history";

/** Load and replay chat history from localStorage on page load. */
function restoreSession() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return;
        const history = JSON.parse(raw);
        history.forEach(({ text, isUser }) => appendMessage(text, isUser, /* persist= */ false));
    } catch (e) {
        localStorage.removeItem(STORAGE_KEY);   // corrupt data — wipe it
    }
}

/** Persist a single message to localStorage history. */
function persistMessage(text, isUser) {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        const history = raw ? JSON.parse(raw) : [];
        history.push({ text, isUser });
        // Keep at most 200 messages to bound localStorage usage
        if (history.length > 200) history.splice(0, history.length - 200);
        localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    } catch (e) {
        // Storage quota exceeded or private browsing — degrade gracefully
    }
}

/** Wipe session (called by Clear History button). */
function clearHistory() {
    localStorage.removeItem(STORAGE_KEY);
    chatArea.innerHTML = "";
    citationsArea.innerHTML = `<p style="text-align:center;color:#64748b;font-size:0.85rem;margin-top:2rem;">
        When the agent orchestrates tools, exact citations and parent context will appear here.</p>`;
}

// ===== UI helpers =====

function setQuery(text) {
    queryInput.value = text;
    queryInput.focus();
}

function handleEnter(e) {
    if (e.key === "Enter" && !e.shiftKey) sendQuery();
}

/**
 * Append a message bubble to the chat area.
 * @param {string} text      Message content (plain for user, Markdown for agent).
 * @param {boolean} isUser   True for user bubbles, false for agent bubbles.
 * @param {boolean} persist  Whether to also save to localStorage (default true).
 */
function appendMessage(text, isUser, persist = true) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${isUser ? "user-msg" : "agent-msg"}`;

    if (isUser) {
        msgDiv.innerText = text;
    } else {
        // Render markdown — vital for matrices, tables, and structured output
        msgDiv.innerHTML = marked.parse(text);
    }

    chatArea.appendChild(msgDiv);
    chatArea.scrollTop = chatArea.scrollHeight;

    if (persist) persistMessage(text, isUser);
}

function updateCitations(citations) {
    citationsArea.innerHTML = "";
    if (!citations || citations.length === 0) {
        citationsArea.innerHTML = `<p style="text-align:center;color:#64748b;font-size:0.85rem;margin-top:2rem;">No citations for this response.</p>`;
        return;
    }

    citations.forEach(c => {
        const card = document.createElement("div");
        card.className = "citation-card";
        card.setAttribute("data-source", c.source_type);

        let displayUrl = c.url;
        if (c.url.startsWith("local://")) displayUrl = "Internal Index";

        card.innerHTML = `
            <div class="citation-header">
                <span>${c.label} ${c.title.substring(0, 35)}${c.title.length > 35 ? "…" : ""}</span>
                <span class="source-tag">${c.source_type.replace(/_/g, " ")}</span>
            </div>
            <div style="font-size:0.75rem;color:#94a3b8;word-break:break-all;">
                <a href="${c.url.startsWith("http") ? c.url : "#"}" target="_blank"
                   style="color:var(--accent);text-decoration:none;">${displayUrl}</a>
            </div>
            <div class="citation-snippet">"${c.snippet}"</div>
        `;
        citationsArea.appendChild(card);
    });
}

// ===== Query submission =====

async function sendQuery() {
    const query = queryInput.value.trim();
    if (!query) return;

    appendMessage(query, true);
    queryInput.value = "";
    sendBtn.disabled = true;
    loader.style.display = "block";

    try {
        const res = await fetch("/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query }),
        });
        const data = await res.json();

        if (res.ok) {
            appendMessage(data.answer, false);
            updateCitations(data.citations);
        } else {
            appendMessage(
                `**Backend Error:**\n\`\`\`text\n${data.detail || "Failed to fetch response"}\n\`\`\``,
                false
            );
        }
    } catch (err) {
        appendMessage(
            `**Network Error:**\nEnsure the backend is running (\`python server.py --local\`).`,
            false
        );
    } finally {
        sendBtn.disabled = false;
        loader.style.display = "none";
        queryInput.focus();
    }
}

// ===== Ingestion with job-status polling =====

async function pollJobStatus(jobId, listItem, filename) {
    const MAX_POLLS = 40;   // 40 × 3 s = 2 min timeout
    for (let i = 0; i < MAX_POLLS; i++) {
        await new Promise(r => setTimeout(r, 3000));
        try {
            const res = await fetch(`/status/${jobId}`);
            if (!res.ok) continue;
            const data = await res.json();
            if (data.status === "success") {
                listItem.innerHTML = `<span>${filename.substring(0, 25)}…</span><span style="color:var(--success)">Indexed ✓</span>`;
                return;
            }
            if (data.status === "failed") {
                listItem.innerHTML = `<span>${filename.substring(0, 25)}…</span><span style="color:#ef4444">Failed: ${data.message || ""}</span>`;
                return;
            }
        } catch (e) { /* network hiccup — keep polling */ }
    }
    // Timed out
    listItem.innerHTML = `<span>${filename.substring(0, 25)}…</span><span style="color:var(--warning)">Timed out</span>`;
}

async function handleFiles(files) {
    if (!files.length) return;

    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const listItem = document.createElement("div");
        listItem.className = "upload-item";
        listItem.innerHTML = `<span>${file.name.substring(0, 25)}…</span><span style="color:var(--warning)">Uploading…</span>`;
        uploadList.insertBefore(listItem, uploadList.firstChild);

        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("/ingest", { method: "POST", body: formData });
            const data = await res.json();
            if (res.ok && data.job_id) {
                listItem.innerHTML = `<span>${file.name.substring(0, 25)}…</span><span style="color:var(--warning)">Ingesting…</span>`;
                // Non-blocking — poll status in background
                pollJobStatus(data.job_id, listItem, file.name);
            } else {
                listItem.innerHTML = `<span>${file.name.substring(0, 25)}…</span><span style="color:#ef4444">Failed</span>`;
            }
        } catch (err) {
            listItem.innerHTML = `<span>${file.name.substring(0, 25)}…</span><span style="color:#ef4444">Network Error</span>`;
        }
    }
}

// ===== File input & drag-and-drop =====

fileInput.addEventListener("change", e => handleFiles(e.target.files));

["dragenter", "dragover", "dragleave", "drop"].forEach(ev =>
    dropArea.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); }, false)
);
dropArea.addEventListener("dragover",  () => dropArea.classList.add("dragover"));
dropArea.addEventListener("dragleave", () => dropArea.classList.remove("dragover"));
dropArea.addEventListener("drop", e => {
    dropArea.classList.remove("dragover");
    handleFiles(e.dataTransfer.files);
});

// ===== Sidebar toggle (mobile) =====

sidebarToggle.addEventListener("click", () => {
    sidebar.classList.toggle("sidebar-open");
});

// Close sidebar when clicking outside it on mobile
document.addEventListener("click", e => {
    if (window.innerWidth <= 768 &&
        !sidebar.contains(e.target) &&
        !sidebarToggle.contains(e.target)) {
        sidebar.classList.remove("sidebar-open");
    }
});

// ===== Boot: restore previous session =====
restoreSession();