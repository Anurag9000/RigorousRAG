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
})();
