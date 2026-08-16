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

  async function request(path) {
    const headers = {};
    const key = sessionStorage.getItem(API_KEY_KEY) || "";
    if (key) headers["X-API-Key"] = key;
    const response = await fetch(path, { headers });
    let payload = null;
    const type = response.headers.get("content-type") || "";
    try {
      payload = type.includes("application/json") ? await response.json() : await response.text();
    } catch { payload = null; }
    if (!response.ok) {
      let detail = payload && typeof payload === "object" ? payload.detail : payload;
      if (detail && typeof detail === "object") detail = detail.message || JSON.stringify(detail);
      throw new Error(detail || `Request failed with status ${response.status}.`);
    }
    return payload;
  }

  function activeSession() {
    try {
      const value = JSON.parse(sessionStorage.getItem(ACTIVE_SESSION_KEY) || "null");
      if (!value || typeof value !== "object") return null;
      if (typeof value.session_id !== "string" || typeof value.project_id !== "string") return null;
      if (!value.session_id || !value.project_id) return null;
      return value;
    } catch {
      return null;
    }
  }

  function params(values) {
    const query = new URLSearchParams();
    Object.entries(values).forEach(([key, value]) => {
      if (value != null && value !== "") query.set(key, String(value));
    });
    const encoded = query.toString();
    return encoded ? `?${encoded}` : "";
  }

  function actionButton(label, handler) {
    const button = node("button", label, "btn small");
    button.type = "button";
    button.addEventListener("click", handler);
    return button;
  }

  function selectTab(button, pane) {
    document.querySelectorAll('[data-tab-group="tools"]').forEach((item) => item.setAttribute("aria-selected", "false"));
    document.querySelectorAll(".right-panel .tool-pane").forEach((item) => item.classList.remove("active"));
    button.setAttribute("aria-selected", "true");
    pane.classList.add("active");
  }

  async function install() {
    const rightPanel = document.querySelector(".right-panel");
    const tabs = rightPanel && rightPanel.querySelector(".tool-tabs");
    if (!rightPanel || !tabs || document.getElementById("tool-integrity")) return;

    const tab = node("button", "Integrity", "tab");
    tab.type = "button";
    tab.dataset.tabGroup = "tools";
    tab.dataset.tab = "integrity";
    tab.setAttribute("aria-selected", "false");
    tabs.appendChild(tab);

    const pane = node("section", null, "tool-pane");
    pane.id = "tool-integrity";
    pane.appendChild(node("h3", "Research integrity"));
    pane.appendChild(node("p", "Verify immutable capsule references against durable server authorities without decrypting replay queries or re-running model providers.", "privacy-note"));

    const toolbar = node("div", null, "section");
    const refresh = actionButton("Refresh integrity", load);
    toolbar.appendChild(refresh);
    pane.appendChild(toolbar);

    const runtimeBox = node("div", null, "section");
    const runtimeStatus = node("div", "", "meta");
    runtimeBox.appendChild(runtimeStatus);
    pane.appendChild(runtimeBox);

    const status = node("div", "", "meta");
    status.style.padding = ".6rem";
    pane.appendChild(status);
    const list = node("div", null, "doc-list");
    pane.appendChild(list);
    rightPanel.appendChild(pane);

    function summaryLine(label, value) {
      return node("div", `${label}: ${value}`, "meta");
    }

    async function verify(capsule, projectId, card) {
      const button = card.querySelector("button[data-integrity-verify]");
      if (button) button.disabled = true;
      try {
        const payload = await request(
          `/research/capsules/${encodeURIComponent(capsule.capsule_id)}/verify${params({ project_id: projectId })}`,
        );
        let details = card.querySelector("[data-integrity-details]");
        if (!details) {
          details = node("div", null, "section");
          details.dataset.integrityDetails = "true";
          card.appendChild(details);
        }
        clear(details);
        details.appendChild(summaryLine("Manifest", payload.manifest_verified ? "verified" : "FAILED"));
        details.appendChild(summaryLine("Evidence", payload.current_evidence ? "current" : "STALE"));
        details.appendChild(summaryLine("Deployment code", payload.code_revision_status || "unknown"));
        details.appendChild(summaryLine("Replay preconditions", payload.replay_preconditions_met ? "met" : "not met"));
        const mismatched = Array.isArray(payload.mismatched_ref_ids) ? payload.mismatched_ref_ids : [];
        const unavailable = Array.isArray(payload.unavailable_ref_ids) ? payload.unavailable_ref_ids : [];
        if (mismatched.length) details.appendChild(summaryLine("Mismatched references", mismatched.join(", ")));
        if (unavailable.length) details.appendChild(summaryLine("Unavailable authorities", unavailable.join(", ")));
        const staleReasons = Array.isArray(payload.stale_reasons) ? payload.stale_reasons : [];
        if (staleReasons.length) {
          details.appendChild(summaryLine(
            "Stale reasons",
            staleReasons.map((item) => String(item.reason || item.event_sha256 || "unknown")).join("; "),
          ));
        }
      } catch (error) {
        status.textContent = `Capsule verification failed: ${error.message}`;
      } finally {
        if (button) button.disabled = false;
      }
    }

    function capsuleCard(capsule, projectId) {
      const card = node("article", null, "doc-card");
      const head = node("div", null, "doc-card-head");
      head.appendChild(node("div", `Capsule ${String(capsule.capsule_id).slice(0, 18)}…`, "doc-title"));
      const verifyButton = actionButton("Verify", () => verify(capsule, projectId, card));
      verifyButton.dataset.integrityVerify = "true";
      head.appendChild(verifyButton);
      card.appendChild(head);
      card.appendChild(summaryLine("Result", capsule.result_id));
      card.appendChild(summaryLine("Created", new Date(Number(capsule.created_at || 0) * 1000).toISOString()));
      card.appendChild(summaryLine("Recorded stale state", capsule.stale ? "STALE" : "current"));
      card.appendChild(summaryLine("Fingerprint", capsule.fingerprint));
      return card;
    }

    async function load() {
      const session = activeSession();
      clear(list);
      if (!session) {
        runtimeStatus.textContent = "No active research session.";
        status.textContent = "Choose an active session in Workspace to inspect its reproducibility capsules.";
        return;
      }
      status.textContent = "Loading runtime and capsule integrity state…";
      try {
        const [runtime, capsules] = await Promise.all([
          request("/research/runtime"),
          request(`/research/capsules${params({ project_id: session.project_id, limit: 500 })}`),
        ]);
        const persistence = runtime && runtime.persistence && typeof runtime.persistence === "object" ? runtime.persistence : {};
        runtimeStatus.textContent = `Persistence ${persistence.metadata_backend || "unknown"} · shared state ${persistence.distributed_shared_state ? "yes" : "no"} · encrypted replay ${persistence.encrypted_replay_configured ? "configured" : "not configured"} · code revision ${persistence.code_revision_configured ? "configured" : "missing"}.`;
        const rows = (Array.isArray(capsules.capsules) ? capsules.capsules : [])
          .filter((item) => item.session_id === session.session_id);
        rows.forEach((capsule) => list.appendChild(capsuleCard(capsule, session.project_id)));
        if (!rows.length) list.appendChild(node("div", "No capsules are bound to the active session yet.", "empty"));
        status.textContent = `${rows.length} capsule${rows.length === 1 ? "" : "s"} available for independent durable-authority verification.`;
      } catch (error) {
        status.textContent = `Integrity state unavailable: ${error.message}`;
      }
    }

    tab.addEventListener("click", () => {
      selectTab(tab, pane);
      load();
    });
    refresh.addEventListener("click", load);
    window.addEventListener("rigorousrag:research-session-changed", () => {
      if (pane.classList.contains("active")) load();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", install, { once: true });
  } else {
    install();
  }
})();
