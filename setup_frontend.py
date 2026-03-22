import os

os.makedirs('frontend', exist_ok=True)

# 1. Advanced HTML with Markdown support and all features
html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RigorousRAG | Enterprise Dashboard</title>
    <!-- Markdown Parser for Comparison Matrices & Debate formatting -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root { --bg: #0f172a; --panel: #1e293b; --text: #e2e8f0; --accent: #3b82f6; --accent-hover: #2563eb; --border: #334155; --success: #10b981; --warning: #f59e0b; }
        body { margin: 0; font-family: system-ui, -apple-system, sans-serif; background-color: var(--bg); color: var(--text); display: flex; height: 100vh; overflow: hidden; }
        .sidebar { width: 380px; background-color: var(--panel); border-right: 1px solid var(--border); display: flex; flex-direction: column; }
        .main { flex: 1; display: flex; flex-direction: column; position: relative; }
        .header { padding: 1rem; border-bottom: 1px solid var(--border); background-color: var(--panel); font-weight: bold; font-size: 1.2rem; display: flex; justify-content: space-between; align-items: center;}
        .status-badge { font-size: 0.75rem; background: var(--success); color: white; padding: 0.2rem 0.5rem; border-radius: 999px; }
        
        /* Chat Area */
        .chat-area { flex: 1; padding: 1.5rem; overflow-y: auto; display: flex; flex-direction: column; gap: 1.5rem; scroll-behavior: smooth; }
        .message { padding: 1.2rem; border-radius: 0.5rem; max-width: 85%; line-height: 1.6; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
        .user-msg { background-color: var(--accent); align-self: flex-end; }
        .agent-msg { background-color: var(--panel); border: 1px solid var(--border); align-self: flex-start; width: 100%; box-sizing: border-box; }
        .agent-msg table { width: 100%; border-collapse: collapse; margin-top: 1rem; margin-bottom: 1rem; }
        .agent-msg th, .agent-msg td { border: 1px solid var(--border); padding: 0.75rem; text-align: left; }
        .agent-msg th { background-color: rgba(255,255,255,0.05); color: var(--accent); }
        .agent-msg pre { background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 0.25rem; overflow-x: auto; }
        
        /* Input Area & Actions */
        .input-area { padding: 1.5rem; background-color: var(--bg); border-top: 1px solid var(--border); }
        .actions-grid { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }
        .action-btn { background: var(--panel); border: 1px solid var(--border); color: var(--text); padding: 0.4rem 0.8rem; border-radius: 0.25rem; cursor: pointer; font-size: 0.8rem; transition: all 0.2s;}
        .action-btn:hover { background: var(--accent); border-color: var(--accent); }
        .input-box { display: flex; gap: 0.5rem; }
        input[type="text"] { flex: 1; padding: 1rem; border-radius: 0.5rem; border: 1px solid var(--border); background: var(--panel); color: var(--text); font-size: 1rem; outline: none; }
        input[type="text"]:focus { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3); }
        button.send { padding: 0 1.5rem; background-color: var(--accent); color: white; border: none; border-radius: 0.5rem; font-weight: bold; cursor: pointer; transition: background 0.2s; }
        button.send:hover { background-color: var(--accent-hover); }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        
        /* Sidebar / Ingestion */
        .upload-zone { padding: 1.5rem; border-bottom: 1px solid var(--border); text-align: center; }
        .drop-area { border: 2px dashed var(--border); border-radius: 0.5rem; padding: 2rem 1rem; cursor: pointer; transition: all 0.2s; background: rgba(255,255,255,0.02); }
        .drop-area:hover, .drop-area.dragover { border-color: var(--success); background: rgba(16, 185, 129, 0.05); }
        .privacy-note { font-size: 0.75rem; color: var(--success); margin-top: 0.75rem; display: flex; align-items: center; justify-content: center; gap: 0.3rem; }
        #file-input { display: none; }
        .upload-list { margin-top: 1rem; font-size: 0.8rem; text-align: left; max-height: 120px; overflow-y: auto; }
        .upload-item { display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid rgba(255,255,255,0.05); }
        
        /* Citations */
        .citations-area { flex: 1; overflow-y: auto; padding: 1rem; background: rgba(0,0,0,0.1); }
        .citation-card { background: var(--bg); border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 0.25rem; padding: 1rem; margin-bottom: 1rem; font-size: 0.85rem; }
        .citation-card[data-source="web_search"] { border-left-color: var(--warning); }
        .citation-card[data-source="handbook"] { border-left-color: #a855f7; }
        .citation-header { font-weight: bold; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: flex-start; gap: 0.5rem;}
        .source-tag { font-size: 0.65rem; padding: 0.1rem 0.4rem; border-radius: 999px; background: rgba(255,255,255,0.1); text-transform: uppercase; white-space: nowrap;}
        .citation-snippet { color: #cbd5e1; font-style: italic; background: rgba(255,255,255,0.03); padding: 0.5rem; border-radius: 0.25rem; margin-top: 0.5rem; }
        .loader { display: none; width: 20px; height: 20px; border: 3px solid var(--border); border-bottom-color: var(--accent); border-radius: 50%; animation: spin 1s linear infinite; margin: 0 1rem; }
        @keyframes spin { 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="header">
            <span>🔬 RigorousRAG</span>
            <span class="status-badge">Local Offline</span>
        </div>
        <div class="upload-zone">
            <div class="drop-area" id="drop-area" onclick="document.getElementById('file-input').click()">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📄</div>
                <p style="margin: 0; font-weight: bold;">Batch Ingest Documents</p>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; color: #94a3b8;">Drop PDF, Word, or TXT files here</p>
                <input type="file" id="file-input" accept=".pdf,.txt,.docx,.md" multiple>
            </div>
            <div class="privacy-note">🔒 Auto-Redaction & Smart Summarization Active</div>
            <div class="upload-list" id="upload-list"></div>
        </div>
        <div class="header" style="font-size: 0.95rem; border-top: 1px solid var(--border); border-bottom: none;">Sources & Grounding (Goal 20)</div>
        <div class="citations-area" id="citations-area">
            <p style="text-align: center; color: #64748b; font-size: 0.85rem; margin-top: 2rem;">When the agent orchestrates tools, exact citations and parent context will appear here.</p>
        </div>
    </div>

    <div class="main">
        <div class="chat-area" id="chat-area">
            <div class="message agent-msg">
                <strong>Engine Ready.</strong><br><br>
                I am connected to the backend Agent. I can orchestrate tools across your Internal Index, the Web, and the Scientific Integrity Suite.<br><br>
                <em>Try uploading some papers, then click the quick actions below!</em>
            </div>
        </div>
        
        <div class="input-area">
            <div class="actions-grid">
                <button class="action-btn" onclick="setQuery('Run a scientific debate on the claim: ')">⚖️ Debate Claim</button>
                <button class="action-btn" onclick="setQuery('Generate a comparison matrix for the following metrics: ')">📊 Comparison Matrix</button>
                <button class="action-btn" onclick="setQuery('Identify conflicts regarding: ')">⚔️ Find Conflicts</button>
                <button class="action-btn" onclick="setQuery('Extract the limitations and disclaimers from the uploaded documents.')">🛡️ Extract Limitations</button>
                <button class="action-btn" onclick="setQuery('Extract the step-by-step wet-lab protocol for: ')">🧪 Extract Protocol</button>
                <button class="action-btn" onclick="setQuery('Search the web for the latest information on: ')">🌐 Web Search</button>
            </div>
            <div class="input-box">
                <input type="text" id="query-input" placeholder="Ask a research question or command the agent..." onkeypress="handleEnter(event)">
                <div style="display: flex; align-items: center; justify-content: center;">
                    <div class="loader" id="loader"></div>
                </div>
                <button class="send" id="send-btn" onclick="sendQuery()">Execute</button>
            </div>
        </div>
    </div>

    <script src="app.js"></script>
</body>
</html>"""

js_content = """
const chatArea = document.getElementById("chat-area");
const queryInput = document.getElementById("query-input");
const sendBtn = document.getElementById("send-btn");
const loader = document.getElementById("loader");
const citationsArea = document.getElementById("citations-area");
const fileInput = document.getElementById("file-input");
const uploadList = document.getElementById("upload-list");
const dropArea = document.getElementById("drop-area");

// Configure marked.js for safe rendering
marked.setOptions({ breaks: true, gfm: true });

function setQuery(text) {
    queryInput.value = text;
    queryInput.focus();
}

function handleEnter(e) {
    if (e.key === "Enter") sendQuery();
}

function appendMessage(text, isUser) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${isUser ? "user-msg" : "agent-msg"}`;
    
    if (isUser) {
        msgDiv.innerText = text;
    } else {
        // Render markdown for agent messages (vital for matrices and structured output)
        msgDiv.innerHTML = marked.parse(text);
    }
    
    chatArea.appendChild(msgDiv);
    chatArea.scrollTop = chatArea.scrollHeight;
}

function updateCitations(citations) {
    citationsArea.innerHTML = "";
    if (!citations || citations.length === 0) {
        citationsArea.innerHTML = "<p style=\\"text-align: center; color: #64748b; font-size: 0.85rem; margin-top: 2rem;\\">No citations for this response.</p>";
        return;
    }
    
    citations.forEach(c => {
        const card = document.createElement("div");
        card.className = "citation-card";
        card.setAttribute("data-source", c.source_type);
        
        // Clean up URL for display
        let displayUrl = c.url;
        if(c.url.startsWith("local://")) displayUrl = "Internal Index";
        
        card.innerHTML = `
            <div class="citation-header">
                <span>${c.label} ${c.title.substring(0, 35)}${c.title.length > 35 ? "..." : ""}</span>
                <span class="source-tag">${c.source_type.replace("_", " ")}</span>
            </div>
            <div style="font-size: 0.75rem; color: #94a3b8; word-break: break-all;">
                <a href="${c.url.startsWith("http") ? c.url : "#"}" target="_blank" style="color: var(--accent); text-decoration: none;">${displayUrl}</a>
            </div>
            <div class="citation-snippet">"${c.snippet}"</div>
        `;
        citationsArea.appendChild(card);
    });
}

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
            body: JSON.stringify({ query: query }) 
        });
        
        const data = await res.json();
        
        if (res.ok) {
            appendMessage(data.answer, false);
            updateCitations(data.citations);
        } else {
            appendMessage(`**Backend Error:**\\n\`\`\`text\\n${data.detail || "Failed to fetch response"}\\n\`\`\``, false);
        }
    } catch (err) {
        appendMessage(`**Network Error:**\\nEnsure the backend server is running (\`python server.py --local\`).`, false);
    } finally {
        sendBtn.disabled = false;
        loader.style.display = "none";
        queryInput.focus();
    }
}

// Advanced Batch File Upload Logic
async function handleFiles(files) {
    if (files.length === 0) return;
    
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const listItem = document.createElement("div");
        listItem.className = "upload-item";
        listItem.innerHTML = `<span>${file.name.substring(0, 25)}...</span><span style="color:var(--warning)">Ingesting...</span>`;
        uploadList.insertBefore(listItem, uploadList.firstChild); // Add to top
        
        const formData = new FormData();
        formData.append("file", file);
        
        try {
            const res = await fetch("/ingest", { method: "POST", body: formData });
            const data = await res.json();
            if (res.ok) {
                listItem.innerHTML = `<span>${file.name.substring(0, 25)}...</span><span style="color:var(--success)">Indexed + Redacted</span>`;
            } else {
                listItem.innerHTML = `<span>${file.name.substring(0, 25)}...</span><span style="color:#ef4444">Failed</span>`;
            }
        } catch (err) {
            listItem.innerHTML = `<span>${file.name.substring(0, 25)}...</span><span style="color:#ef4444">Network Error</span>`;
        }
    }
}

fileInput.addEventListener("change", (e) => handleFiles(e.target.files));

// Drag and drop setup
["dragenter", "dragover", "dragleave", "drop"].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
});
function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

dropArea.addEventListener("dragover", () => dropArea.classList.add("dragover"));
dropArea.addEventListener("dragleave", () => dropArea.classList.remove("dragover"));
dropArea.addEventListener("drop", (e) => {
    dropArea.classList.remove("dragover");
    handleFiles(e.dataTransfer.files);
});
"""

with open("frontend/index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

with open("frontend/app.js", "w", encoding="utf-8") as f:
    f.write(js_content)

# Safely update server.py to mount the frontend
with open("server.py", "r", encoding="utf-8") as f:
    server_code = f.read()

if "from fastapi.staticfiles import StaticFiles" not in server_code:
    # 1. Add Import
    server_code = server_code.replace(
        "from fastapi import FastAPI, UploadFile",
        "from fastapi import FastAPI, UploadFile\\nfrom fastapi.staticfiles import StaticFiles"
    )
    
    # 2. Remove old root route to prevent collision
    server_code = server_code.replace(
        "@app.get(\\"/\\")\\nasync def root():\\n    return {\\"message\\": \\"Academic Search Engine API is running.\\"}",
        "# Web Frontend replaces API root\\n"
    )
    
    # 3. Mount frontend before uvicorn run
    server_code = server_code.replace(
        "if __name__ == \\"__main__\\":",
        "app.mount(\\"/\\", StaticFiles(directory=\\"frontend\\", html=True), name=\\"static\\")\\n\\nif __name__ == \\"__main__\\":"
    )

    with open("server.py", "w", encoding="utf-8") as f:
        f.write(server_code)

print("Advanced Frontend setup complete! It is now fully integrated with all backend features.")
