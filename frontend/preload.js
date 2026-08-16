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

  function loadExtension(src, name) {
    const script = document.createElement("script");
    script.src = src;
    script.defer = true;
    script.dataset.rigorousragExtension = name;
    document.head.appendChild(script);
  }

  loadExtension("/static/research_query.js", "research-query-persistence");
  loadExtension("/static/research.js", "research-workspace");
  loadExtension("/static/integrity.js", "research-integrity");
})();
