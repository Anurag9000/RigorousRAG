"use strict";

(() => {
  const API_KEY_KEY = "rigorousrag_session_api_key";
  const ACTIVE_SESSION_KEY = "rigorousrag_active_research_session_v1";

  function node(tag, text, className = "") {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text != null) element.textContent = String(text);
    return element;
  }

  function clear(element) {
    while (element.firstChild) element.removeChild(element.firstChild);
  }

  async function request(path, options = {}) {
    const headers = { ...(options.headers || {}) };
    const key = sessionStorage.getItem(API_KEY_KEY) || "";
    if (key) headers["X-API-Key"] = key;
    const response = await fetch(path, { ...options, headers });
    let payload = null;
    const type = response.headers.get("content-type") || "";
    try {
      payload = type.includes("application/json") ? await response.json() : await response.text();
    } catch { payload = null; }
    if (!response.ok) {
      const detail = payload && typeof payload === "object" ? payload.detail : payload;
      const error = new Error(detail || `Request failed with status ${response.status}.`);
      error.status = response.status;
      throw error;
    }
    return payload;
  }

  function activeSession() {
    try {
      const value = JSON.parse(sessionStorage.getItem(ACTIVE_SESSION_KEY) || "null");
      if (!value || typeof value !== "object") return null;
      if (typeof value.session_id !== "string" || typeof value.fingerprint !== "string") return null;
      return value;
    } catch {
      sessionStorage.removeItem(ACTIVE_SESSION_KEY);
      return null;
    }
  }

  function setActiveSession(session, project) {
    if (!session) {
      sessionStorage.removeItem(ACTIVE_SESSION_KEY);
      window.dispatchEvent(new CustomEvent("rigorousrag:research-session-changed", { detail: null }));
      return;
    }
    const value = {
      session_id: String(session.session_id || ""),
      fingerprint: String(session.fingerprint || ""),
      project_id: String(project && project.project_id || session.project_id || ""),
      project_title: String(project && project.title || ""),
      closed_at: session.closed_at || null,
    };
    if (!value.session_id || !/^[0-9a-f]{64}$/i.test(value.fingerprint) || value.closed_at) return;
    sessionStorage.setItem(ACTIVE_SESSION_KEY, JSON.stringify(value));
    window.dispatchEvent(new CustomEvent("rigorousrag:research-session-changed", { detail: value }));
  }

  function selectWorkspace(button, pane) {
    document.querySelectorAll('[data-tab-group="tools"]').forEach((item) => item.setAttribute("aria-selected", "false"));
    document.querySelectorAll(".right-panel .tool-pane").forEach((item) => item.classList.remove("active"));
    button.setAttribute("aria-selected", "true");
    pane.classList.add("active");
  }

  function field(label, id, { multiline = false, placeholder = "" } = {}) {
    const wrapper = document.createElement("div");
    wrapper.appendChild(node("label", label, "label"));
    const input = document.createElement(multiline ? "textarea" : "input");
    input.className = "field";
    input.id = id;
    input.placeholder = placeholder;
    input.maxLength = multiline ? 20000 : 1000;
    if (!multiline) input.type = "text";
    wrapper.appendChild(input);
    return wrapper;
  }

  function projectCard(project, onOpen) {
    const card = node("article", null, "doc-card");
    const head = node("div", null, "doc-card-head");
    head.appendChild(node("div", project.title || project.project_id, "doc-title"));
    const open = node("button", "Open", "btn small");
    open.type = "button";
    open.addEventListener("click", () => onOpen(project));
    head.appendChild(open);
    card.appendChild(head);
    card.appendChild(node("div", project.research_question || "", "summary"));
    card.appendChild(node("div", `Project ${project.project_id}`, "meta"));
    card.appendChild(node("div", `Corpora ${(project.corpora || []).length} · ${(project.tags || []).join(", ") || "no tags"}`, "meta"));
    return card;
  }

  async function install() {
    const rightPanel = document.querySelector(".right-panel");
    const tabs = rightPanel && rightPanel.querySelector(".tool-tabs");
    if (!rightPanel || !tabs || document.getElementById("tool-workspace")) return;

    const button = node("button", "Workspace", "tab");
    button.type = "button";
    button.dataset.tabGroup = "tools";
    button.dataset.tab = "workspace";
    button.setAttribute("aria-selected", "false");
    tabs.appendChild(button);

    const pane = node("section", null, "tool-pane");
    pane.id = "tool-workspace";
    pane.appendChild(node("h3", "Research workspace"));
    pane.appendChild(node("p", "Persist projects, authoritative result IDs and server-owned citation fingerprints without copying private evidence into browser history.", "privacy-note"));

    const activeBox = node("div", null, "section");
    const activeLabel = node("div", "No active research session. Chat results will still be stored when the research API is available.", "meta");
    const clearActive = node("button", "Use stateless research", "btn small");
    clearActive.type = "button";
    clearActive.style.marginTop = ".4rem";
    clearActive.addEventListener("click", () => {
      setActiveSession(null, null);
      renderActive();
    });
    activeBox.appendChild(activeLabel);
    activeBox.appendChild(clearActive);
    pane.appendChild(activeBox);

    const createBox = node("div", null, "section");
    createBox.appendChild(field("Project title", "workspace-title", { placeholder: "Systematic review of …" }));
    createBox.appendChild(field("Research question", "workspace-question", { multiline: true, placeholder: "What evidence answers …?" }));
    createBox.appendChild(field("Tags", "workspace-tags", { placeholder: "climate, retrieval, methods" }));
    const createButton = node("button", "Create project", "btn primary");
    createButton.type = "button";
    createButton.style.marginTop = ".6rem";
    createBox.appendChild(createButton);
    pane.appendChild(createBox);

    const toolbar = node("div", null, "section");
    const refreshButton = node("button", "Refresh projects", "btn small");
    refreshButton.type = "button";
    toolbar.appendChild(refreshButton);
    pane.appendChild(toolbar);

    const status = node("div", "", "meta");
    status.style.padding = ".6rem";
    pane.appendChild(status);
    const projects = node("div", null, "doc-list");
    pane.appendChild(projects);

    const sessionBox = node("section", null, "section");
    sessionBox.hidden = true;
    const sessionHeading = node("h4", "Sessions");
    sessionBox.appendChild(sessionHeading);
    const newSession = node("button", "New session", "btn small");
    newSession.type = "button";
    sessionBox.appendChild(newSession);
    const sessions = node("div", null, "doc-list");
    sessions.style.padding = ".6rem 0 0";
    sessionBox.appendChild(sessions);
    pane.appendChild(sessionBox);
    rightPanel.appendChild(pane);

    let activeProject = null;

    function renderActive() {
      const current = activeSession();
      if (!current) {
        activeLabel.textContent = "No active research session. Chat results will still be stored when the research API is available.";
        clearActive.disabled = true;
        return;
      }
      activeLabel.textContent = `Active session: ${current.session_id}${current.project_title ? ` · ${current.project_title}` : ""}`;
      clearActive.disabled = false;
    }

    async function closeSession(session, project) {
      try {
        const updated = await request(`/research/sessions/${encodeURIComponent(session.session_id)}/close`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ expected_session_fingerprint: session.fingerprint }),
        });
        const current = activeSession();
        if (current && current.session_id === session.session_id) setActiveSession(null, null);
        renderActive();
        await loadSessions(project);
        return updated;
      } catch (error) {
        status.textContent = `Could not close session: ${error.message}`;
        return null;
      }
    }

    async function loadSessions(project) {
      activeProject = project;
      sessionBox.hidden = false;
      sessionHeading.textContent = `Sessions · ${project.title || project.project_id}`;
      clear(sessions);
      sessions.appendChild(node("div", "Loading sessions…", "empty"));
      try {
        const payload = await request(`/research/projects/${encodeURIComponent(project.project_id)}/sessions`);
        clear(sessions);
        const rows = Array.isArray(payload.sessions) ? payload.sessions : [];
        if (!rows.length) {
          sessions.appendChild(node("div", "No sessions yet.", "empty"));
          return;
        }
        const current = activeSession();
        for (const session of rows) {
          if (current && current.session_id === session.session_id && !session.closed_at && current.fingerprint !== session.fingerprint) {
            setActiveSession(session, project);
          }
          const card = node("article", null, "doc-card");
          const head = node("div", null, "doc-card-head");
          head.appendChild(node("div", session.session_id, "doc-title"));
          const actions = node("div", null, "doc-actions");
          if (!session.closed_at) {
            const use = node("button", "Use", "btn small");
            use.type = "button";
            use.addEventListener("click", () => {
              setActiveSession(session, project);
              renderActive();
              status.textContent = `Active session set to ${session.session_id}.`;
            });
            actions.appendChild(use);
            const close = node("button", "Close", "btn small danger");
            close.type = "button";
            close.addEventListener("click", () => closeSession(session, project));
            actions.appendChild(close);
          }
          head.appendChild(actions);
          card.appendChild(head);
          card.appendChild(node("div", `${(session.turns || []).length} turns · ${session.closed_at ? "closed" : "open"}`, "meta"));
          card.appendChild(node("div", `Fingerprint ${session.fingerprint}`, "meta"));
          sessions.appendChild(card);
        }
      } catch (error) {
        clear(sessions);
        sessions.appendChild(node("div", `Could not load sessions: ${error.message}`, "empty"));
      }
    }

    async function loadProjects() {
      status.textContent = "Loading projects…";
      clear(projects);
      try {
        const payload = await request("/research/projects?limit=200");
        const rows = Array.isArray(payload.projects) ? payload.projects : [];
        for (const project of rows) projects.appendChild(projectCard(project, loadSessions));
        if (!rows.length) projects.appendChild(node("div", "No research projects yet.", "empty"));
        status.textContent = `${rows.length} project${rows.length === 1 ? "" : "s"}`;
      } catch (error) {
        status.textContent = `Workspace unavailable: ${error.message}`;
      }
    }

    button.addEventListener("click", () => {
      selectWorkspace(button, pane);
      renderActive();
      loadProjects();
    });
    refreshButton.addEventListener("click", loadProjects);
    createButton.addEventListener("click", async () => {
      const title = document.getElementById("workspace-title").value.trim();
      const researchQuestion = document.getElementById("workspace-question").value.trim();
      const tags = document.getElementById("workspace-tags").value.split(",").map((item) => item.trim()).filter(Boolean).slice(0, 100);
      if (!title || !researchQuestion) {
        status.textContent = "Project title and research question are required.";
        return;
      }
      createButton.disabled = true;
      try {
        await request("/research/projects", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title, research_question: researchQuestion, corpora: [], tags }),
        });
        document.getElementById("workspace-title").value = "";
        document.getElementById("workspace-question").value = "";
        document.getElementById("workspace-tags").value = "";
        await loadProjects();
      } catch (error) {
        status.textContent = `Could not create project: ${error.message}`;
      } finally {
        createButton.disabled = false;
      }
    });
    newSession.addEventListener("click", async () => {
      if (!activeProject) return;
      newSession.disabled = true;
      try {
        const created = await request(`/research/projects/${encodeURIComponent(activeProject.project_id)}/sessions`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
        });
        setActiveSession(created, activeProject);
        renderActive();
        await loadSessions(activeProject);
      } catch (error) {
        status.textContent = `Could not create session: ${error.message}`;
      } finally {
        newSession.disabled = false;
      }
    });

    renderActive();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", install, { once: true });
  } else {
    install();
  }
})();
