"use strict";

(() => {
  const API_KEY_KEY = "rigorousrag_session_api_key";

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
      throw new Error(detail || `Request failed with status ${response.status}.`);
    }
    return payload;
  }

  function selectWorkspace(button, pane) {
    document.querySelectorAll('[data-tab-group="right"]').forEach((item) => item.setAttribute("aria-selected", "false"));
    document.querySelectorAll(".right-panel .tool-pane").forEach((item) => item.classList.remove("active"));
    button.setAttribute("aria-selected", "true");
    pane.classList.add("active");
  }

  function field(label, id, { multiline = false, placeholder = "" } = {}) {
    const wrapper = node("label", null, "field");
    wrapper.appendChild(node("span", label));
    const input = document.createElement(multiline ? "textarea" : "input");
    input.id = id;
    input.placeholder = placeholder;
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
    const tabs = document.querySelector(".right-panel .tool-tabs");
    const body = document.querySelector(".right-panel .tool-body");
    if (!tabs || !body || document.getElementById("tool-workspace")) return;

    const button = node("button", "Workspace", "tool-tab");
    button.type = "button";
    button.dataset.tabGroup = "right";
    button.dataset.tab = "workspace";
    button.setAttribute("aria-selected", "false");
    tabs.appendChild(button);

    const pane = node("section", null, "tool-pane");
    pane.id = "tool-workspace";
    pane.appendChild(node("h3", "Research workspace"));
    pane.appendChild(node("p", "Persist projects and research-session fingerprints without copying private evidence into browser history.", "meta"));

    const createBox = node("div", null, "tool-form");
    createBox.appendChild(field("Project title", "workspace-title", { placeholder: "Systematic review of …" }));
    createBox.appendChild(field("Research question", "workspace-question", { multiline: true, placeholder: "What evidence answers …?" }));
    createBox.appendChild(field("Tags", "workspace-tags", { placeholder: "climate, retrieval, methods" }));
    const createButton = node("button", "Create project", "btn primary");
    createButton.type = "button";
    createBox.appendChild(createButton);
    pane.appendChild(createBox);

    const toolbar = node("div", "", "tool-actions");
    const refreshButton = node("button", "Refresh projects", "btn small");
    refreshButton.type = "button";
    toolbar.appendChild(refreshButton);
    pane.appendChild(toolbar);

    const status = node("div", "", "meta");
    pane.appendChild(status);
    const projects = node("div", null, "doc-list");
    pane.appendChild(projects);

    const sessionBox = node("section", null, "tool-form");
    sessionBox.hidden = true;
    const sessionHeading = node("h4", "Sessions");
    sessionBox.appendChild(sessionHeading);
    const sessionActions = node("div", null, "tool-actions");
    const newSession = node("button", "New session", "btn small");
    newSession.type = "button";
    sessionActions.appendChild(newSession);
    sessionBox.appendChild(sessionActions);
    const sessions = node("div", null, "doc-list");
    sessionBox.appendChild(sessions);
    pane.appendChild(sessionBox);
    body.appendChild(pane);

    let activeProject = null;

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
        for (const session of rows) {
          const card = node("article", null, "doc-card");
          card.appendChild(node("div", session.session_id, "doc-title"));
          card.appendChild(node("div", `${(session.turns || []).length} turns · ${session.closed_at ? "closed" : "open"}`, "meta"));
          card.title = `Session fingerprint ${session.fingerprint}`;
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
        await request(`/research/projects/${encodeURIComponent(activeProject.project_id)}/sessions`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
        });
        await loadSessions(activeProject);
      } catch (error) {
        status.textContent = `Could not create session: ${error.message}`;
      } finally {
        newSession.disabled = false;
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", install, { once: true });
  } else {
    install();
  }
})();
