"use strict";

(() => {
  const nativeFetch = window.fetch.bind(window);
  const DEFAULT_PRELOAD_TIMEOUT_MS = 135_000;

  window.fetch = function fetchWithInitialDeadline(input, init = {}) {
    const options = init && typeof init === "object" ? { ...init } : {};
    if (options.signal) return nativeFetch(input, options);

    const controller = new AbortController();
    options.signal = controller.signal;
    const timer = window.setTimeout(
      () => controller.abort(),
      DEFAULT_PRELOAD_TIMEOUT_MS,
    );
    return nativeFetch(input, options).finally(() => {
      window.clearTimeout(timer);
    });
  };

  // Optional research-workspace extension. It owns its own safe-DOM bindings so it can
  // be delivered independently from the core chat bundle without inline script policy.
  const researchScript = document.createElement("script");
  researchScript.src = "/static/research.js";
  researchScript.defer = true;
  researchScript.dataset.rigorousragExtension = "research-workspace";
  document.head.appendChild(researchScript);
})();