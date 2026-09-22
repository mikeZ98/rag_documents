/* --------------------------------------------------------------------------
 * RAG Showcase — chat client.
 *
 * Consumes POST /rag/stream. EventSource cannot issue POST requests, so the
 * SSE frames are parsed by hand from the fetch ReadableStream: split on blank
 * lines, read `event:` and `data:` fields, ignore comments (keep-alive pings).
 *
 * No framework, no build step, no npm — one CDN script (marked) for markdown.
 * -------------------------------------------------------------------------- */
(() => {
  "use strict";

  const thread = document.getElementById("thread");
  const emptyState = document.getElementById("empty-state");
  const form = document.getElementById("composer");
  const input = document.getElementById("query");
  const kSelect = document.getElementById("k");
  const sendButton = document.getElementById("send");
  const stopButton = document.getElementById("stop");
  const counter = document.getElementById("counter");
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");

  const MIN_QUERY = 3;
  const MAX_QUERY = 500;

  let controller = null;

  /* ----------------------------- utilities ------------------------------ */

  /** Session id groups a conversation's traces in Langfuse. */
  const sessionId = (() => {
    const fresh = `web-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
    try {
      const stored = sessionStorage.getItem("rag-session-id");
      if (stored) return stored;
      sessionStorage.setItem("rag-session-id", fresh);
    } catch {
      /* private mode / blocked storage: a per-load id is fine */
    }
    return fresh;
  })();

  const escapeHtml = (value) =>
    value.replace(/[&<>"']/g, (char) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    })[char]);

  /**
   * Render markdown safely: the text is derived from user-supplied PDFs, so
   * HTML is escaped *before* parsing. Markdown itself still renders fully.
   */
  const renderMarkdown = (text) => {
    const escaped = escapeHtml(text);
    if (window.marked && typeof window.marked.parse === "function") {
      return window.marked.parse(escaped, { breaks: true, gfm: true });
    }
    return `<p>${escaped.replace(/\n/g, "<br>")}</p>`;
  };

  const formatMs = (value) =>
    value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`;

  const scrollToBottom = () => {
    thread.scrollTop = thread.scrollHeight;
  };

  const element = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  /* ------------------------------- health -------------------------------- */

  async function refreshHealth() {
    try {
      const response = await fetch("/health", { headers: { accept: "application/json" } });
      const body = await response.json();
      const ready = body.status === "ok";
      statusDot.className = `dot ${ready ? "ok" : "degraded"}`;
      statusText.textContent = `${body.llm_model} · ${body.indexed_points ?? 0} chunks indexed`;
      statusText.title = [
        `provider: ${body.llm_provider} (${body.llm_endpoint})`,
        `collection: ${body.collection}`,
        `embedding: ${body.embedding_model}`,
        `reranker: ${body.reranker_model}`,
        body.detail ? `note: ${body.detail}` : "",
      ]
        .filter(Boolean)
        .join("\n");
    } catch {
      statusDot.className = "dot down";
      statusText.textContent = "service unreachable";
    }
  }

  /* ------------------------------ rendering ------------------------------ */

  function renderSources(container, payload) {
    const details = element("details", "sources");
    const summary = element("summary", "sources-summary");
    summary.append(element("span", "chevron", "›"));
    summary.append(element("span", null, "Retrieved Sources & Context"));
    summary.append(element("span", "badge-count", String(payload.sources.length)));
    summary.append(
      element("span", null, `· reranked from ${payload.candidates_considered} candidates`),
    );
    details.append(summary);

    const body = element("div", "sources-body");
    payload.sources.forEach((source, index) => {
      const row = element("div", "source");
      row.append(element("span", "source-index", `[${index + 1}]`));

      const main = element("div");
      const head = element("div", "source-head");
      head.append(element("span", "source-file", source.source));
      head.append(element("span", "source-page", `page ${source.page}`));
      const rerank = element("span", "score rerank", `rerank ${source.rerank_score.toFixed(3)}`);
      rerank.title = "Cross-encoder relevance score (stage 2)";
      head.append(rerank);
      const dense = element("span", "score", `dense ${source.dense_score.toFixed(3)}`);
      dense.title = "Cosine similarity from Qdrant (stage 1)";
      head.append(dense);
      main.append(head);
      main.append(element("p", "source-preview", source.preview));

      row.append(main);
      body.append(row);
    });
    details.append(body);
    container.prepend(details);
    return details;
  }

  function renderMetrics(container, metrics) {
    const bar = element("div", "metrics");
    const add = (label, value) => {
      const item = element("span");
      item.append(document.createTextNode(`${label} `));
      item.append(element("b", null, value));
      bar.append(item);
    };
    add("retrieval", formatMs(metrics.retrievalMs));
    if (metrics.ttftMs !== null) add("TTFT", formatMs(metrics.ttftMs));
    add("total", formatMs(metrics.totalMs));
    if (metrics.usage) {
      add("tokens", `${metrics.usage.input_tokens} in / ${metrics.usage.output_tokens} out`);
    }
    container.append(bar);
    return bar;
  }

  /* ---------------------------- SSE plumbing ----------------------------- */

  /** Parse one SSE frame into {event, data}; returns null for comments. */
  function parseFrame(frame) {
    let event = "message";
    const data = [];
    for (const line of frame.split("\n")) {
      if (!line || line.startsWith(":")) continue; // keep-alive ping
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).trim());
    }
    if (data.length === 0) return null;
    return { event, data: data.join("\n") };
  }

  async function* readEvents(response, signal) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    try {
      while (!signal.aborted) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
        let split = buffer.indexOf("\n\n");
        while (split >= 0) {
          const frame = parseFrame(buffer.slice(0, split));
          buffer = buffer.slice(split + 2);
          if (frame) yield frame;
          split = buffer.indexOf("\n\n");
        }
      }
    } finally {
      reader.cancel().catch(() => {});
    }
  }

  /* ------------------------------ the turn ------------------------------- */

  async function ask(question) {
    if (emptyState) emptyState.remove();

    const turn = element("div", "turn");
    turn.append(element("div", "bubble-user", question));
    thread.append(turn);

    const answerTurn = element("div", "turn");
    const answer = element("div", "answer");
    const markdown = element("div", "markdown");
    const cursor = element("span", "cursor");
    markdown.append(cursor);
    answer.append(markdown);
    answerTurn.append(answer);
    thread.append(answerTurn);
    scrollToBottom();

    controller = new AbortController();
    setBusy(true);

    const started = performance.now();
    let ttft = null;
    let retrievalMs = 0;
    let usage = null;
    let text = "";
    let finished = false;

    const paint = () => {
      markdown.innerHTML = renderMarkdown(text);
      markdown.append(cursor);
      scrollToBottom();
    };

    try {
      const response = await fetch("/rag/stream", {
        method: "POST",
        headers: { "content-type": "application/json", accept: "text/event-stream" },
        body: JSON.stringify({
          query: question,
          k: Number(kSelect.value),
          session_id: sessionId,
        }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        const detail = await response
          .json()
          .then((body) => body.detail || body.error)
          .catch(() => `HTTP ${response.status}`);
        cursor.remove();
        answer.append(element("div", "error", detail));
        return;
      }

      for await (const frame of readEvents(response, controller.signal)) {
        const payload = JSON.parse(frame.data);
        if (frame.event === "sources") {
          retrievalMs = payload.retrieval_ms;
          renderSources(answer, payload);
          if (payload.trace_id) answer.dataset.traceId = payload.trace_id;
          scrollToBottom();
        } else if (frame.event === "token") {
          if (ttft === null) ttft = performance.now() - started;
          text += payload.text;
          paint();
        } else if (frame.event === "done") {
          usage = payload.usage;
          finished = true;
        } else if (frame.event === "error") {
          cursor.remove();
          answer.append(element("div", "error", payload.detail));
          finished = true;
        }
      }
    } catch (error) {
      cursor.remove();
      if (error.name === "AbortError") {
        answer.append(element("div", "metrics", "stopped"));
        return;
      }
      answer.append(element("div", "error", `Connection failed: ${error.message}`));
      return;
    } finally {
      cursor.remove();
      if (text) paint();
      setBusy(false);
      controller = null;
    }

    if (finished || text) {
      renderMetrics(answer, {
        retrievalMs,
        ttftMs: ttft,
        totalMs: performance.now() - started,
        usage,
      });
      scrollToBottom();
    }
  }

  /* ------------------------------- wiring -------------------------------- */

  function setBusy(busy) {
    sendButton.disabled = busy;
    stopButton.hidden = !busy;
    input.disabled = busy;
    if (!busy) input.focus();
  }

  function autoResize() {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
    counter.textContent = String(input.value.trim().length);
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (question.length < MIN_QUERY || question.length > MAX_QUERY || sendButton.disabled) return;
    input.value = "";
    autoResize();
    ask(question);
  });

  input.addEventListener("input", autoResize);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  stopButton.addEventListener("click", () => controller && controller.abort());

  document.getElementById("suggestions")?.addEventListener("click", (event) => {
    const question = event.target.dataset?.q;
    if (!question) return;
    input.value = question;
    autoResize();
    form.requestSubmit();
  });

  autoResize();
  input.focus();
  refreshHealth();
  setInterval(refreshHealth, 30000);
})();
