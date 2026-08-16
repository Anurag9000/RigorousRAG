"use strict";

(() => {
  const API_KEY_KEY = "rigorousrag_session_api_key";
  const ACTIVE_SESSION_KEY = "rigorousrag_active_research_session_v1";
  const MAX_EXPORT_BYTES = 8 * 1024 * 1024;

  function node(tag, text, className = "") {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text != null) element.textContent = String(text);
    return element;
  }

  function clear(element) {
    while (element.firstChild) element.removeChild(element.firstChild);
  }

  function actionButton(label, handler, className = "btn small") {
    const button = node("button", label, className);
    button.type = "button";
    button.addEventListener("click", handler);
    return button;
  }

  function authHeaders(extra = {}) {
    const headers = { ...extra };
    const key = sessionStorage.getItem(API_KEY_KEY) || "";
    if (key) headers["X-API-Key"] = key;
    return headers;
  }

  async function request(path, options = {}) {
    const response = await fetch(path, {
      ...options,
      headers: authHeaders(options.headers || {}),
    });
    const type = response.headers.get("content-type") || "";
    let payload = null;
    try {
      payload = type.includes("application/json") ? await response.json() : await response.text();
    } catch {
      payload = null;
    }
    if (!response.ok) {
      let detail = payload && typeof payload === "object" ? payload.detail : payload;
      if (detail && typeof detail === "object") detail = detail.message || JSON.stringify(detail);
      throw new Error(detail || `Request failed with status ${response.status}.`);
    }
    return payload;
  }

  function activeProjectId() {
    try {
      const value = JSON.parse(sessionStorage.getItem(ACTIVE_SESSION_KEY) || "null");
      return value && typeof value.project_id === "string" ? value.project_id : "";
    } catch {
      return "";
    }
  }

  function selectedPermissions(statusPayload) {
    const raw = statusPayload && statusPayload.project && Array.isArray(statusPayload.project.permissions)
      ? statusPayload.project.permissions
      : [];
    return new Set(raw.map((value) => String(value)));
  }

  function hasPermission(statusPayload, permission) {
    return selectedPermissions(statusPayload).has(permission);
  }

  function safeFilename(value, fallback) {
    const cleaned = String(value || "").replace(/[^A-Za-z0-9._-]+/g, "_").slice(0, 160);
    return cleaned || fallback;
  }

  async function downloadAuthenticated(path, filename) {
    const response = await fetch(path, { headers: authHeaders() });
    if (!response.ok) {
      let detail = "";
      try { detail = await response.text(); } catch { detail = ""; }
      throw new Error(detail || `Export failed with status ${response.status}.`);
    }
    const length = Number(response.headers.get("content-length") || "0");
    if (Number.isFinite(length) && length > MAX_EXPORT_BYTES) {
      throw new Error("Export exceeds the browser safety limit.");
    }
    const blob = await response.blob();
    if (blob.size > MAX_EXPORT_BYTES) throw new Error("Export exceeds the browser safety limit.");
    const url = URL.createObjectURL(blob);
    try {
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      link.rel = "noopener";
      document.body.appendChild(link);
      link.click();
      link.remove();
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  function selectPane(button, pane) {
    document.querySelectorAll('[data-tab-group="tools"]').forEach((item) => item.setAttribute("aria-selected", "false"));
    document.querySelectorAll(".right-panel .tool-pane").forEach((item) => item.classList.remove("active"));
    button.setAttribute("aria-selected", "true");
    pane.classList.add("active");
  }

  function summaryLines(summary) {
    if (!summary || typeof summary !== "object") return [];
    const keys = [
      "complete", "executable", "model_type", "scope", "anchor_node_id", "target_node_id",
      "variable", "node_count", "reach_count", "record_count", "object_count", "row_count",
      "time_window_count", "diagnostic_count", "unresolved_count", "source_fingerprint",
      "topology_fingerprint", "package_fingerprint", "plan_fingerprint", "index_fingerprint",
      "projection_fingerprint",
    ];
    const lines = [];
    keys.forEach((key) => {
      if (!(key in summary)) return;
      const value = Array.isArray(summary[key]) ? summary[key].join(", ") : summary[key];
      lines.push(`${key}: ${String(value)}`);
    });
    if (summary.summary && typeof summary.summary === "object") {
      Object.entries(summary.summary).slice(0, 20).forEach(([key, value]) => lines.push(`${key}: ${String(value)}`));
    }
    return lines;
  }

  async function install() {
    const rightPanel = document.querySelector(".right-panel");
    const tabs = rightPanel && rightPanel.querySelector(".tool-tabs");
    if (!rightPanel || !tabs || document.getElementById("tool-hydrology")) return;

    const tab = node("button", "Hydrology", "tab");
    tab.type = "button";
    tab.dataset.tabGroup = "tools";
    tab.dataset.tab = "hydrology";
    tab.setAttribute("aria-selected", "false");
    tabs.appendChild(tab);

    const pane = node("section", null, "tool-pane");
    pane.id = "tool-hydrology";
    pane.appendChild(node("h3", "Governed hydrology evidence"));
    pane.appendChild(node(
      "p",
      "Inspect server-owned HEC/HMS topology, engineering packages, retrieval plans, evidence projections and deterministic reports. Browser code never parses local engineering files or decides artifact freshness.",
      "privacy-note",
    ));

    const projectSection = node("div", null, "section");
    const projectLabel = node("label", "Research project", "label");
    const projectSelect = document.createElement("select");
    projectSelect.className = "field";
    projectSelect.appendChild(new Option("Select a project…", ""));
    projectSection.appendChild(projectLabel);
    projectSection.appendChild(projectSelect);

    const controls = node("div", null, "doc-actions");
    const refreshProjects = actionButton("Refresh projects", loadProjects);
    const refreshStatus = actionButton("Refresh hydrology", loadStatus);
    controls.appendChild(refreshProjects);
    controls.appendChild(refreshStatus);
    projectSection.appendChild(controls);

    const historyLabel = node("label", null, "meta");
    const historyToggle = document.createElement("input");
    historyToggle.type = "checkbox";
    historyToggle.style.marginRight = ".35rem";
    historyLabel.appendChild(historyToggle);
    historyLabel.appendChild(document.createTextNode("Show artifact history"));
    projectSection.appendChild(historyLabel);
    pane.appendChild(projectSection);

    const status = node("div", "Select a research project.", "meta");
    status.style.padding = ".6rem";
    pane.appendChild(status);

    const artifacts = node("div", null, "doc-list");
    pane.appendChild(artifacts);

    const detailSection = node("section", null, "section");
    detailSection.hidden = true;
    const detailHeading = node("h4", "Artifact details");
    const detailMeta = node("div", "", "meta");
    const detailActions = node("div", null, "doc-actions");
    const detailBody = node("pre", "", "summary");
    detailBody.style.whiteSpace = "pre-wrap";
    detailBody.style.maxHeight = "28rem";
    detailBody.style.overflow = "auto";
    detailSection.appendChild(detailHeading);
    detailSection.appendChild(detailMeta);
    detailSection.appendChild(detailActions);
    detailSection.appendChild(detailBody);
    pane.appendChild(detailSection);

    rightPanel.appendChild(pane);

    let currentStatus = null;
    let currentDetail = null;

    tab.addEventListener("click", () => {
      selectPane(tab, pane);
      if (!projectSelect.options.length || projectSelect.options.length === 1) loadProjects();
    });
    projectSelect.addEventListener("change", loadStatus);
    historyToggle.addEventListener("change", loadStatus);
    window.addEventListener("rigorousrag:research-session-changed", () => {
      const projectId = activeProjectId();
      if (projectId && Array.from(projectSelect.options).some((item) => item.value === projectId)) {
        projectSelect.value = projectId;
        loadStatus();
      }
    });

    async function loadProjects() {
      refreshProjects.disabled = true;
      try {
        const payload = await request("/research/projects");
        const rows = Array.isArray(payload && payload.projects) ? payload.projects : [];
        const desired = activeProjectId() || projectSelect.value;
        clear(projectSelect);
        projectSelect.appendChild(new Option("Select a project…", ""));
        rows.forEach((project) => {
          const option = new Option(project.title || project.project_id, project.project_id);
          option.dataset.role = String(project.access && project.access.role || "");
          projectSelect.appendChild(option);
        });
        if (desired && rows.some((project) => project.project_id === desired)) {
          projectSelect.value = desired;
          await loadStatus();
        } else {
          clear(artifacts);
          detailSection.hidden = true;
          status.textContent = rows.length ? "Select a research project." : "No accessible research projects.";
        }
      } catch (error) {
        status.textContent = `Could not load projects: ${error.message}`;
      } finally {
        refreshProjects.disabled = false;
      }
    }

    async function loadStatus() {
      const projectId = projectSelect.value;
      currentStatus = null;
      currentDetail = null;
      clear(artifacts);
      detailSection.hidden = true;
      if (!projectId) {
        status.textContent = "Select a research project.";
        return;
      }
      refreshStatus.disabled = true;
      status.textContent = "Loading hydrology artifacts…";
      try {
        const suffix = historyToggle.checked ? "?include_history=true&limit=1000" : "?limit=1000";
        const payload = await request(`/research/projects/${encodeURIComponent(projectId)}/hydrology/status${suffix}`);
        currentStatus = payload;
        renderStatus(payload);
      } catch (error) {
        status.textContent = `Could not load hydrology status: ${error.message}`;
      } finally {
        refreshStatus.disabled = false;
      }
    }

    function renderStatus(payload) {
      clear(artifacts);
      const rows = Array.isArray(payload && payload.artifacts) ? payload.artifacts : [];
      const project = payload && payload.project || {};
      status.textContent = `${project.title || project.project_id || "Project"} · ${project.role || "unknown role"} · ${rows.length} hydrology artifact${rows.length === 1 ? "" : "s"} · stale ledger ${payload.stale_state_complete ? "complete" : "truncated/unknown"}`;
      if (!rows.length) {
        artifacts.appendChild(node("div", "No governed hydrology artifacts have been persisted for this project.", "empty"));
        return;
      }
      rows.forEach((artifact) => {
        const card = node("article", null, "doc-card");
        const head = node("div", null, "doc-card-head");
        head.appendChild(node("div", `${artifact.kind} · ${artifact.logical_id}`, "doc-title"));
        head.appendChild(actionButton("Inspect", () => inspectArtifact(artifact)));
        card.appendChild(head);
        const staleText = artifact.stale === true ? "STALE" : artifact.stale === false ? "current" : "freshness unknown";
        card.appendChild(node("div", `v${artifact.version} · ${artifact.is_current ? "current generation" : "historical"} · ${staleText}`, "meta"));
        card.appendChild(node("div", `Fingerprint ${artifact.fingerprint}`, "meta"));
        summaryLines(artifact.summary).slice(0, 12).forEach((line) => card.appendChild(node("div", line, "meta")));
        if (Array.isArray(artifact.stale_reasons)) {
          artifact.stale_reasons.slice(0, 3).forEach((reason) => card.appendChild(node("div", `Stale: ${reason.reason || reason.event_sha256 || "dependency changed"}`, "meta")));
        }
        artifacts.appendChild(card);
      });
    }

    async function inspectArtifact(artifact) {
      const projectId = projectSelect.value;
      if (!projectId) return;
      detailSection.hidden = false;
      clear(detailActions);
      detailHeading.textContent = `${artifact.kind} · ${artifact.logical_id}`;
      detailMeta.textContent = "Loading typed server projection…";
      detailBody.textContent = "";
      try {
        const query = historyToggle.checked && !artifact.is_current
          ? `?fingerprint=${encodeURIComponent(artifact.fingerprint)}&detail_limit=1000`
          : "?detail_limit=1000";
        const payload = await request(`/research/projects/${encodeURIComponent(projectId)}/hydrology/status/${encodeURIComponent(artifact.kind)}/${encodeURIComponent(artifact.logical_id)}${query}`);
        currentDetail = payload;
        detailMeta.textContent = `${payload.stale === true ? "STALE" : payload.stale === false ? "current" : "freshness unknown"} · ${payload.fingerprint}`;
        detailBody.textContent = JSON.stringify(payload.details || {}, null, 2);
        addDetailActions(artifact, payload);
      } catch (error) {
        detailMeta.textContent = `Could not inspect artifact: ${error.message}`;
      }
    }

    function addDetailActions(artifact, payload) {
      clear(detailActions);
      if (!currentStatus) return;
      const projectId = projectSelect.value;
      if (artifact.kind === "evidence_projection" && artifact.is_current && payload.stale === false && hasPermission(currentStatus, "report.write")) {
        detailActions.appendChild(actionButton("Create report", async () => {
          const proposed = `${artifact.logical_id}-report`;
          const reportId = window.prompt("Hydrology report ID", proposed);
          if (!reportId) return;
          try {
            const created = await request(`/research/projects/${encodeURIComponent(projectId)}/hydrology/reports`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ report_id: reportId.trim(), projection_id: artifact.logical_id }),
            });
            status.textContent = `Hydrology report ${created.report_id} created at fingerprint ${created.fingerprint}.`;
            await loadStatus();
          } catch (error) {
            status.textContent = `Could not create hydrology report: ${error.message}`;
          }
        }, "btn small primary"));
      }
      if (artifact.kind === "evidence_report" && hasPermission(currentStatus, "report.read")) {
        const base = safeFilename(artifact.logical_id, "hydrology-report");
        detailActions.appendChild(actionButton("Export Markdown", async () => {
          try {
            await downloadAuthenticated(
              `/research/projects/${encodeURIComponent(projectId)}/hydrology/reports/${encodeURIComponent(artifact.logical_id)}/markdown`,
              `${base}.md`,
            );
          } catch (error) {
            status.textContent = `Could not export Markdown: ${error.message}`;
          }
        }));
        detailActions.appendChild(actionButton("Export CSV", async () => {
          try {
            await downloadAuthenticated(
              `/research/projects/${encodeURIComponent(projectId)}/hydrology/reports/${encodeURIComponent(artifact.logical_id)}/csv`,
              `${base}.csv`,
            );
          } catch (error) {
            status.textContent = `Could not export CSV: ${error.message}`;
          }
        }));
      }
    }

    await loadProjects();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => install(), { once: true });
  } else {
    install();
  }
})();
