const transcript = document.querySelector("#transcript");
const form = document.querySelector("#chatForm");
const input = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");
const statusPill = document.querySelector("#status");
const sessionInput = document.querySelector("#sessionId");
const newSessionButton = document.querySelector("#newSession");
const clearSessionButton = document.querySelector("#clearSession");
const searchWebInput = document.querySelector("#searchWeb");
const audienceSelect = document.querySelector("#audience");

const SESSION_KEY = "agentic_legal_session_id";

function createSessionId() {
  if (window.crypto && window.crypto.randomUUID) {
    return `sess_${window.crypto.randomUUID()}`;
  }
  return `sess_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function getSessionId() {
  let sessionId = window.localStorage.getItem(SESSION_KEY);
  if (!sessionId) {
    sessionId = createSessionId();
    window.localStorage.setItem(SESSION_KEY, sessionId);
  }
  return sessionId;
}

function setStatus(text, state = "") {
  statusPill.textContent = text;
  statusPill.className = `status-pill ${state}`.trim();
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderInlineMarkdown(value) {
  const protectedSegments = [];
  let output = value
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, (_, label, url) => {
      const index = protectedSegments.length;
      protectedSegments.push(`<a href="${url}" target="_blank" rel="noreferrer">${label}</a>`);
      return `%%SEGMENT${index}%%`;
    })
    .replace(/`([^`]+)`/g, (_, code) => {
      const index = protectedSegments.length;
      protectedSegments.push(`<code>${code}</code>`);
      return `%%SEGMENT${index}%%`;
    })
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/__([^_]+)__/g, "<strong>$1</strong>")
    .replace(/\*([^*\n]+)\*/g, "<em>$1</em>")
    .replace(/_([^_\n]+)_/g, "<em>$1</em>");

  protectedSegments.forEach((segment, index) => {
    output = output.replaceAll(`%%SEGMENT${index}%%`, segment);
  });
  return output;
}

function renderMarkdownBlock(block, codeBlocks) {
  const codeMatch = block.match(/^%%CODE_BLOCK_(\d+)%%$/);
  if (codeMatch) {
    return codeBlocks[Number(codeMatch[1])] || "";
  }

  const lines = block.split("\n").filter((line) => line.trim().length > 0);
  const heading = block.match(/^(#{1,3})\s+(.+)$/);
  if (heading) {
    const level = Math.min(heading[1].length + 2, 5);
    return `<h${level}>${renderInlineMarkdown(heading[2].trim())}</h${level}>`;
  }

  if (lines.length > 0 && lines.every((line) => /^[-*]\s+/.test(line.trim()))) {
    const items = lines
      .map((line) => `<li>${renderInlineMarkdown(line.trim().replace(/^[-*]\s+/, ""))}</li>`)
      .join("");
    return `<ul>${items}</ul>`;
  }

  if (lines.length > 0 && lines.every((line) => /^\d+\.\s+/.test(line.trim()))) {
    const items = lines
      .map((line) => `<li>${renderInlineMarkdown(line.trim().replace(/^\d+\.\s+/, ""))}</li>`)
      .join("");
    return `<ol>${items}</ol>`;
  }

  return `<p>${renderInlineMarkdown(block).replace(/\n/g, "<br>")}</p>`;
}

function renderMarkdown(value) {
  const codeBlocks = [];
  let source = escapeHtml(value || "").replace(/\r\n/g, "\n").trim();
  if (!source) {
    return "";
  }

  source = source.replace(/```[^\n]*\n([\s\S]*?)```/g, (_, code) => {
    const index = codeBlocks.length;
    codeBlocks.push(`<pre><code>${code.trim()}</code></pre>`);
    return `\n\n%%CODE_BLOCK_${index}%%\n\n`;
  });

  return source
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean)
    .map((block) => renderMarkdownBlock(block, codeBlocks))
    .join("");
}

function scrollToBottom() {
  transcript.scrollTop = transcript.scrollHeight;
}

function renderEmpty() {
  transcript.innerHTML = '<div class="empty-state">Chưa có hội thoại trong session này. Nhập câu hỏi để bắt đầu.</div>';
}

function renderUserMessage(message) {
  const turn = document.createElement("article");
  turn.className = "turn user";
  turn.innerHTML = `<div class="bubble">${escapeHtml(message)}</div>`;
  transcript.appendChild(turn);
}

function renderAssistantMessage(response) {
  const turn = document.createElement("article");
  turn.className = "turn assistant";

  const citations = (response.citations || [])
    .map((citation, index) => {
      const label = citation.article
        ? `${citation.doc_title} - Điều ${citation.article}${citation.clause ? `, khoản ${citation.clause}` : ""}`
        : citation.doc_title;
      const safeLabel = escapeHtml(label || `Nguồn ${index + 1}`);
      if (citation.source_url) {
        return `<span class="citation"><a href="${escapeHtml(citation.source_url)}" target="_blank" rel="noreferrer">${safeLabel}</a></span>`;
      }
      return `<span class="citation">${safeLabel}</span>`;
    })
    .join("");

  const followUps = (response.follow_up_questions || [])
    .map((question) => `<button class="follow-up" type="button">${escapeHtml(question)}</button>`)
    .join("");

  turn.innerHTML = `
    <div class="bubble markdown">${renderMarkdown(response.answer || "")}</div>
    <div class="meta-row">
      <span class="meta-chip">Mode: ${escapeHtml(response.mode || "fallback")}</span>
      <span class="meta-chip">Confidence: ${Math.round((response.confidence || 0) * 100)}%</span>
      <span class="meta-chip">Trace: ${escapeHtml(response.trace_id || "n/a")}</span>
    </div>
    ${citations ? `<div class="citation-list">${citations}</div>` : ""}
    ${response.disclaimer ? `<div class="disclaimer">${escapeHtml(response.disclaimer)}</div>` : ""}
    ${followUps ? `<div class="follow-ups">${followUps}</div>` : ""}
  `;
  transcript.appendChild(turn);
}

function renderHistory(turns) {
  transcript.innerHTML = "";
  if (!turns.length) {
    renderEmpty();
    return;
  }

  for (const turn of turns) {
    renderUserMessage(turn.user_message);
    renderAssistantMessage({
      answer: turn.assistant_message,
      confidence: 0,
      mode: "faq",
      citations: turn.citations || [],
      disclaimer: "",
      follow_up_questions: [],
      trace_id: turn.trace_id,
    });
  }
  scrollToBottom();
}

async function loadHistory() {
  const sessionId = sessionInput.value.trim();
  if (!sessionId) {
    renderEmpty();
    return;
  }

  setStatus("Đang tải...", "loading");
  try {
    const response = await fetch(`/v1/chat/sessions/${encodeURIComponent(sessionId)}`);
    if (!response.ok) {
      throw new Error(`Không tải được session (${response.status})`);
    }
    const data = await response.json();
    renderHistory(data.turns || []);
    setStatus("Sẵn sàng");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function sendMessage(message) {
  sendButton.disabled = true;
  input.disabled = true;
  setStatus("Đang trả lời...", "loading");

  if (transcript.querySelector(".empty-state")) {
    transcript.innerHTML = "";
  }
  renderUserMessage(message);
  scrollToBottom();

  try {
    const response = await fetch("/v1/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionInput.value.trim(),
        message,
        search_web: searchWebInput.checked,
        user_profile: {
          audience: audienceSelect.value,
          locale: "vi-VN",
        },
      }),
    });

    if (!response.ok) {
      const body = await response.text();
      throw new Error(body || `Yêu cầu thất bại (${response.status})`);
    }

    const data = await response.json();
    renderAssistantMessage(data);
    setStatus("Sẵn sàng");
  } catch (error) {
    renderAssistantMessage({
      answer: `Không thể hoàn tất yêu cầu: ${error.message}`,
      confidence: 0,
      mode: "fallback",
      citations: [],
      disclaimer: "",
      follow_up_questions: [],
      trace_id: "error",
    });
    setStatus("Có lỗi", "error");
  } finally {
    sendButton.disabled = false;
    input.disabled = false;
    input.focus();
    scrollToBottom();
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (message.length < 3) {
    input.focus();
    return;
  }
  input.value = "";
  await sendMessage(message);
});

newSessionButton.addEventListener("click", () => {
  const sessionId = createSessionId();
  sessionInput.value = sessionId;
  window.localStorage.setItem(SESSION_KEY, sessionId);
  renderEmpty();
  setStatus("Session mới");
  input.focus();
});

clearSessionButton.addEventListener("click", async () => {
  const sessionId = sessionInput.value.trim();
  if (!sessionId) {
    return;
  }

  setStatus("Đang xóa...", "loading");
  try {
    const response = await fetch(`/v1/chat/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
    if (!response.ok) {
      throw new Error(`Không xóa được session (${response.status})`);
    }
    renderEmpty();
    setStatus("Đã xóa");
  } catch (error) {
    setStatus(error.message, "error");
  }
});

sessionInput.addEventListener("change", () => {
  const sessionId = sessionInput.value.trim() || createSessionId();
  sessionInput.value = sessionId;
  window.localStorage.setItem(SESSION_KEY, sessionId);
  loadHistory();
});

transcript.addEventListener("click", (event) => {
  const target = event.target;
  if (target instanceof HTMLButtonElement && target.classList.contains("follow-up")) {
    input.value = target.textContent || "";
    input.focus();
  }
});

sessionInput.value = getSessionId();
loadHistory();
