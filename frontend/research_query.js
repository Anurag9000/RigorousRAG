"use strict";

(() => {
  const ACTIVE_SESSION_KEY = "rigorousrag_active_research_session_v1";
  const LAST_RESULT_KEY = "rigorousrag_last_research_result_v1";
  const wrappedFetch = window.fetch.bind(window);

  function activeSession() {
    try {
      const value = JSON.parse(sessionStorage.getItem(ACTIVE_SESSION_KEY) || "null");
      if (!value || typeof value !== "object") return null;
      if (typeof value.session_id !== "string" || typeof value.fingerprint !== "string") return null;
      if (!/^[0-9a-f]{64}$/i.test(value.fingerprint)) return null;
      return value;
    } catch {
      sessionStorage.removeItem(ACTIVE_SESSION_KEY);
      return null;
    }
  }

  async function researchFetch(input, init = {}) {
    const rawUrl = typeof input === "string" ? input : (input && input.url ? input.url : "");
    let parsed = null;
    try { parsed = new URL(rawUrl, window.location.origin); } catch { parsed = null; }
    if (!parsed || parsed.origin !== window.location.origin || parsed.pathname !== "/query") {
      return wrappedFetch(input, init);
    }

    const method = String(init && init.method || "GET").toUpperCase();
    if (method !== "POST") return wrappedFetch(input, init);
    let body = null;
    try { body = JSON.parse(String(init.body || "{}")); } catch { body = null; }
    if (!body || typeof body !== "object" || typeof body.query !== "string") {
      return wrappedFetch(input, init);
    }

    const session = activeSession();
    const researchBody = {
      query: body.query,
      model: body.model || null,
      notes: "",
      session_id: session ? session.session_id : null,
      expected_session_fingerprint: session ? session.fingerprint : null,
    };
    const researchInit = {
      ...init,
      headers: { ...(init.headers || {}), "Content-Type": "application/json" },
      body: JSON.stringify(researchBody),
    };
    const response = await wrappedFetch("/research/query", researchInit);
    if (response.status === 404 || response.status === 405) {
      return wrappedFetch(input, init);
    }
    if (response.ok) {
      try {
        const payload = await response.clone().json();
        if (payload && typeof payload === "object") {
          if (typeof payload.result_id === "string") {
            sessionStorage.setItem(
              LAST_RESULT_KEY,
              JSON.stringify({ result_id: payload.result_id, created_at: payload.created_at || Date.now() / 1000 }),
            );
          }
          if (session && typeof payload.session_fingerprint === "string" && /^[0-9a-f]{64}$/i.test(payload.session_fingerprint)) {
            const updated = { ...session, fingerprint: payload.session_fingerprint };
            sessionStorage.setItem(ACTIVE_SESSION_KEY, JSON.stringify(updated));
            window.dispatchEvent(new CustomEvent("rigorousrag:research-session-changed", { detail: updated }));
          }
        }
      } catch { /* response remains authoritative even if browser metadata parsing fails */ }
    }
    return response;
  }

  window.fetch = researchFetch;
})();
