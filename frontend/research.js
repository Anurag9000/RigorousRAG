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

  function permissionSet(value) {
    const raw = value && value.access && Array.isArray(value.access.permissions) ? value.access.permissions : [];
    return new Set(raw.map((item) => String(item)));
  }

  function hasPermission(value, permission) {
    const permissions = permissionSet(value);
    return !value || !value.access || permissions.has(permission);
  }

  function params(values) {
    const query = new URLSearchParams();
    Object.entries(values).forEach(([key, value]) => {
      if (value != null && value !== "") query.set(key, String(value));
    });
    const encoded = query.toString();
    return encoded ? `?${encoded}` : "";
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
      let detail = payload && typeof payload === "object" ? payload.detail : payload;
      if (detail && typeof detail === "object") detail = detail.message || JSON.stringify(detail);
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

  function field(label, id, { multiline = false, placeholder = "", type = "text" } = {}) {
    const wrapper = document.createElement("div");
    wrapper.appendChild(node("label", label, "label"));
    const input = document.createElement(multiline ? "textarea" : "input");
    input.className = "field";
    input.id = id;
    input.placeholder = placeholder;
    input.maxLength = multiline ? 20000 : 1000;
    if (!multiline) input.type = type;
    wrapper.appendChild(input);
    return wrapper;
  }

  function accessMeta(value) {
    if (!value || !value.access) return "owner-local";
    const role = String(value.access.role || "unknown");
    return `${role}${value.access.shared ? " · shared" : " · owner"}`;
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
    card.appendChild(node("div", `Access ${accessMeta(project)}`, "meta"));
    card.appendChild(node("div", `Corpora ${(project.corpora || []).length} · ${(project.tags || []).join(", ") || "no tags"}`, "meta"));
    return card;
  }

  function actionButton(label, handler, className = "btn small") {
    const button = node("button", label, className);
    button.type = "button";
    button.addEventListener("click", handler);
    return button;
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
    pane.appendChild(node("p", "Projects, sessions, result IDs, replay status, reports and reproducibility capsules stay server-authoritative. Private evidence and replay queries are not copied into browser history.", "privacy-note"));

    const activeBox = node("div", null, "section");
    const activeLabel = node("div", "No active research session.", "meta");
    const clearActive = actionButton("Use stateless research", () => {
      setActiveSession(null, null);
      renderActive();
    });
    clearActive.style.marginTop = ".4rem";
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
    const refreshButton = actionButton("Refresh projects", loadProjects);
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
    const newSession = actionButton("New session", createSession);
    sessionBox.appendChild(newSession);
    const sessions = node("div", null, "doc-list");
    sessions.style.padding = ".6rem 0 0";
    sessionBox.appendChild(sessions);
    pane.appendChild(sessionBox);

    const aclBox = node("section", null, "section");
    aclBox.hidden = true;
    aclBox.appendChild(node("h4", "Project access"));
    const aclList = node("div", null, "doc-list");
    aclBox.appendChild(aclList);
    const aclForm = node("div", null, "section");
    aclForm.appendChild(field("Principal ID", "workspace-acl-principal", { placeholder: "collaborator identity" }));
    const roleLabel = node("label", "Role", "label");
    const roleSelect = document.createElement("select");
    roleSelect.className = "field";
    roleSelect.id = "workspace-acl-role";
    ["viewer", "reviewer", "editor"].forEach((role) => {
      const option = document.createElement("option");
      option.value = role;
      option.textContent = role;
      roleSelect.appendChild(option);
    });
    aclForm.appendChild(roleLabel);
    aclForm.appendChild(roleSelect);
    aclForm.appendChild(field("Expires at (optional ISO timestamp)", "workspace-acl-expires", { placeholder: "2026-12-31T23:59:59Z" }));
    const grantButton = actionButton("Grant / update access", grantAccess, "btn primary");
    grantButton.style.marginTop = ".6rem";
    aclForm.appendChild(grantButton);
    aclBox.appendChild(aclForm);
    pane.appendChild(aclBox);

    const artifactBox = node("section", null, "section");
    artifactBox.hidden = true;
    const artifactHeading = node("h4", "Session evidence lifecycle");
    artifactBox.appendChild(artifactHeading);
    const artifactStatus = node("div", "", "meta");
    artifactBox.appendChild(artifactStatus);
    const artifacts = node("div", null, "doc-list");
    artifactBox.appendChild(artifacts);
    pane.appendChild(artifactBox);

    rightPanel.appendChild(pane);

    let activeProject = null;
    let inspectedSession = null;

    function renderActive() {
      const current = activeSession();
      if (!current) {
        activeLabel.textContent = "No active research session. Stateless research remains available.";
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
      inspectedSession = null;
      artifactBox.hidden = true;
      sessionBox.hidden = false;
      sessionHeading.textContent = `Sessions · ${project.title || project.project_id}`;
      newSession.disabled = !hasPermission(project, "session.write");
      clear(sessions);
      sessions.appendChild(node("div", "Loading sessions…", "empty"));
      await loadACL(project);
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
          const inspect = actionButton("Inspect", () => loadArtifacts(project, session));
          actions.appendChild(inspect);
          if (!session.closed_at && hasPermission(session, "research.execute")) {
            actions.appendChild(actionButton("Use", () => {
              setActiveSession(session, project);
              renderActive();
              status.textContent = `Active session set to ${session.session_id}.`;
            }));
          }
          if (!session.closed_at && hasPermission(session, "session.write")) {
            actions.appendChild(actionButton("Close", () => closeSession(session, project), "btn small danger"));
          }
          head.appendChild(actions);
          card.appendChild(head);
          card.appendChild(node("div", `${(session.turns || []).length} turns · ${session.closed_at ? "closed" : "open"}`, "meta"));
          card.appendChild(node("div", `Access ${accessMeta(session)}`, "meta"));
          card.appendChild(node("div", `Fingerprint ${session.fingerprint}`, "meta"));
          sessions.appendChild(card);
        }
      } catch (error) {
        clear(sessions);
        sessions.appendChild(node("div", `Could not load sessions: ${error.message}`, "empty"));
      }
    }

    async function loadACL(project) {
      const canManage = hasPermission(project, "acl.manage");
      aclBox.hidden = !canManage;
      if (!canManage) return;
      clear(aclList);
      aclList.appendChild(node("div", "Loading grants…", "empty"));
      try {
        const payload = await request(`/research/projects/${encodeURIComponent(project.project_id)}/acl`);
        clear(aclList);
        const grants = Array.isArray(payload.grants) ? payload.grants : [];
        grants.forEach((grant) => {
          const card = node("article", null, "doc-card");
          const head = node("div", null, "doc-card-head");
          head.appendChild(node("div", `${grant.principal_id} · ${grant.role}`, "doc-title"));
          if (grant.role !== "owner") {
            head.appendChild(actionButton("Revoke", async () => {
              try {
                await request(`/research/projects/${encodeURIComponent(project.project_id)}/acl/${encodeURIComponent(grant.principal_id)}`, { method: "DELETE" });
                await loadACL(project);
              } catch (error) {
                status.textContent = `Could not revoke access: ${error.message}`;
              }
            }, "btn small danger"));
          }
          card.appendChild(head);
          card.appendChild(node("div", `Permissions ${(grant.permissions || []).join(", ")}`, "meta"));
          card.appendChild(node("div", grant.expires_at ? `Expires ${new Date(grant.expires_at * 1000).toISOString()}` : "No expiration", "meta"));
          aclList.appendChild(card);
        });
      } catch (error) {
        clear(aclList);
        aclList.appendChild(node("div", `Could not load project grants: ${error.message}`, "empty"));
      }
    }

    async function grantAccess() {
      if (!activeProject) return;
      const principal = document.getElementById("workspace-acl-principal").value.trim();
      const role = document.getElementById("workspace-acl-role").value;
      const expiresText = document.getElementById("workspace-acl-expires").value.trim();
      if (!principal) {
        status.textContent = "Principal ID is required.";
        return;
      }
      let expiresAt = null;
      if (expiresText) {
        const milliseconds = Date.parse(expiresText);
        if (!Number.isFinite(milliseconds)) {
          status.textContent = "Expiration must be a valid ISO timestamp.";
          return;
        }
        expiresAt = milliseconds / 1000;
      }
      try {
        await request(`/research/projects/${encodeURIComponent(activeProject.project_id)}/acl`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ principal_id: principal, role, expires_at: expiresAt }),
        });
        document.getElementById("workspace-acl-principal").value = "";
        document.getElementById("workspace-acl-expires").value = "";
        await loadACL(activeProject);
      } catch (error) {
        status.textContent = `Could not grant access: ${error.message}`;
      }
    }

    function resultCard(project, session, result, replayByResult, capsulesByResult, reportsByResult) {
      const card = node("article", null, "doc-card");
      const head = node("div", null, "doc-card-head");
      head.appendChild(node("div", `Result ${String(result.result_id).slice(0, 12)}…`, "doc-title"));
      const actions = node("div", null, "doc-actions");
      actions.appendChild(actionButton("History", () => showHistory(session, result)));
      if (hasPermission(session, "capsule.write")) {
        actions.appendChild(actionButton("Capsule preflight", () => preflightCapsule(project, session, result)));
        actions.appendChild(actionButton("Archive capsule", () => createCapsule(project, session, result)));
      }
      if (hasPermission(session, "report.write")) {
        actions.appendChild(actionButton("Create report", () => createReport(project, session, result)));
      }
      head.appendChild(actions);
      card.appendChild(head);
      card.appendChild(node("div", `Strategy ${result.strategy || "unknown"} · model ${result.model || "unknown"}`, "meta"));
      card.appendChild(node("div", `${result.citation_count || 0} citations · ${result.stale ? "STALE" : "current"}`, "meta"));
      card.appendChild(node("div", `Replay recipe ${replayByResult.has(result.result_id) ? "available" : "not available"}`, "meta"));
      card.appendChild(node("div", `Capsules ${(capsulesByResult.get(result.result_id) || []).length} · reports ${(reportsByResult.get(result.result_id) || []).length}`, "meta"));
      return card;
    }

    async function loadArtifacts(project, session) {
      inspectedSession = session;
      artifactBox.hidden = false;
      artifactHeading.textContent = `Session evidence lifecycle · ${session.session_id}`;
      artifactStatus.textContent = "Loading results, replay status, reports and capsules…";
      clear(artifacts);
      try {
        const [resultPayload, replayPayload, capsulePayload, reportPayload] = await Promise.all([
          request(`/research/results${params({ session_id: session.session_id, limit: 100 })}`),
          request(`/research/replay${params({ session_id: session.session_id, limit: 100 })}`),
          request(`/research/capsules${params({ project_id: project.project_id, limit: 200 })}`),
          request(`/research/reports${params({ project_id: project.project_id, limit: 200 })}`),
        ]);
        const results = Array.isArray(resultPayload.results) ? resultPayload.results : [];
        const replayByResult = new Set((replayPayload.recipes || []).map((item) => item.result_id));
        const capsulesByResult = new Map();
        (capsulePayload.capsules || []).filter((item) => item.session_id === session.session_id).forEach((item) => {
          const rows = capsulesByResult.get(item.result_id) || [];
          rows.push(item);
          capsulesByResult.set(item.result_id, rows);
        });
        const reportsByResult = new Map();
        (reportPayload.reports || []).forEach((item) => {
          const rows = reportsByResult.get(item.result_id) || [];
          rows.push(item);
          reportsByResult.set(item.result_id, rows);
        });
        results.forEach((result) => artifacts.appendChild(resultCard(project, session, result, replayByResult, capsulesByResult, reportsByResult)));
        if (!results.length) artifacts.appendChild(node("div", "No finalized results are bound to this session yet.", "empty"));
        artifactStatus.textContent = `${results.length} result${results.length === 1 ? "" : "s"} · ${(capsulePayload.capsules || []).length} project capsule${(capsulePayload.capsules || []).length === 1 ? "" : "s"} · ${(reportPayload.reports || []).length} project report${(reportPayload.reports || []).length === 1 ? "" : "s"}`;
      } catch (error) {
        artifactStatus.textContent = `Could not load session artifacts: ${error.message}`;
      }
    }

    async function showHistory(session, result) {
      try {
        const payload = await request(`/research/results/${encodeURIComponent(result.result_id)}/history${params({ session_id: session.session_id })}`);
        const transitions = Array.isArray(payload.transitions) ? payload.transitions : [];
        artifactStatus.textContent = transitions.length
          ? `Answer history: ${payload.version_count} versions; latest ${String(payload.current.result_id).slice(0, 12)}…; ${transitions.filter((item) => item.answer_changed).length} answer-text change(s).`
          : "Answer history: this result has not been superseded.";
      } catch (error) {
        artifactStatus.textContent = `Could not load answer history: ${error.message}`;
      }
    }

    async function preflightCapsule(project, session, result) {
      try {
        const payload = await request("/research/capsules/preflight", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ project_id: project.project_id, session_id: session.session_id, result_id: result.result_id, require_replay_ready: true }),
        });
        const blockers = [...(payload.blockers || []), ...(payload.replay_blockers || [])];
        artifactStatus.textContent = `Capsule preflight: manifest ${payload.manifest_ready ? "ready" : "blocked"}; replay ${payload.replay_ready ? "ready" : "blocked"}${blockers.length ? ` · ${blockers.join(", ")}` : ""}.`;
      } catch (error) {
        artifactStatus.textContent = `Capsule preflight failed: ${error.message}`;
      }
    }

    async function createCapsule(project, session, result) {
      try {
        const payload = await request("/research/capsules", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ project_id: project.project_id, session_id: session.session_id, result_id: result.result_id, require_replay_ready: false }),
        });
        artifactStatus.textContent = `Immutable capsule ${payload.capsule_id} created with fingerprint ${payload.fingerprint}.`;
        await loadArtifacts(project, session);
      } catch (error) {
        artifactStatus.textContent = `Could not create capsule: ${error.message}`;
      }
    }

    async function createReport(project, session, result) {
      try {
        const payload = await request("/research/reports", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ project_id: project.project_id, session_id: session.session_id, result_id: result.result_id }),
        });
        artifactStatus.textContent = `Report ${payload.report_id} created from authoritative result ${result.result_id}.`;
        await loadArtifacts(project, session);
      } catch (error) {
        artifactStatus.textContent = `Could not create report: ${error.message}`;
      }
    }

    async function loadProjects() {
      status.textContent = "Loading projects…";
      clear(projects);
      try {
        const payload = await request("/research/projects?limit=200");
        const rows = Array.isArray(payload.projects) ? payload.projects : [];
        rows.forEach((project) => projects.appendChild(projectCard(project, loadSessions)));
        if (!rows.length) projects.appendChild(node("div", "No research projects yet.", "empty"));
        status.textContent = `${rows.length} project${rows.length === 1 ? "" : "s"}`;
      } catch (error) {
        status.textContent = `Workspace unavailable: ${error.message}`;
      }
    }

    async function createSession() {
      if (!activeProject) return;
      newSession.disabled = true;
      try {
        const created = await request(`/research/projects/${encodeURIComponent(activeProject.project_id)}/sessions`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
        });
        if (hasPermission(created, "research.execute")) setActiveSession(created, activeProject);
        renderActive();
        await loadSessions(activeProject);
      } catch (error) {
        status.textContent = `Could not create session: ${error.message}`;
      } finally {
        newSession.disabled = !activeProject || !hasPermission(activeProject, "session.write");
      }
    }

    button.addEventListener("click", () => {
      selectWorkspace(button, pane);
      renderActive();
      loadProjects();
    });
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

    renderActive();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", install, { once: true });
  } else {
    install();
  }
})();
