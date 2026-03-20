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
        citationsArea.innerHTML = '<p style="text-align: center; color: #64748b; font-size: 0.85rem; margin-top: 2rem;">No citations for this response.</p>';
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
                <span class="source-tag">${c.source_type.replace(/_/g, " ")}</span>
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
            appendMessage(`**Backend Error:**\n\`\`\`text\n${data.detail || "Failed to fetch response"}\n\`\`\``, false);
        }
    } catch (err) {
        appendMessage(`**Network Error:**\nEnsure the backend server is running (\`python server.py --local\`).`, false);
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