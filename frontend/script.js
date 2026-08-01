/**
 * script.js — ForgeMind AI Frontend
 * ===================================
 * Pure ES6 Vanilla JavaScript. No frameworks, no libraries.
 * Uses fetch() with async/await throughout.
 *
 * Communicates with FastAPI backend at BACKEND_URL.
 * Endpoints used:
 *   GET  /          → health / welcome check
 *   POST /chat      → send a message, receive AI answer
 *   POST /upload    → upload a PDF file
 *   GET  /documents → list indexed PDFs
 *
 * Author: ForgeMind AI Team
 */

'use strict';

/* ============================================================
   CONFIGURATION
   ============================================================ */

/** Change this if your backend runs on a different host/port */
const BACKEND_URL = 'https://forgemind-ai-production-25e8.up.railway.app';

/* ============================================================
   DOM REFERENCES
   All element lookups happen once here so we avoid repeated
   querySelector calls inside tight loops or event handlers.
   ============================================================ */
const dom = {
  // Status
  statusDot:          document.getElementById('status-dot'),
  statusLabel:        document.getElementById('status-label'),

  // Welcome / Messages
  welcomeScreen:      document.getElementById('welcome-screen'),
  messages:           document.getElementById('messages'),
  thinkingRow:        document.getElementById('thinking-row'),

  // Input
  messageInput:       document.getElementById('message-input'),
  sendBtn:            document.getElementById('send-btn'),

  // Sidebar
  sidebarUploadZone:  document.getElementById('sidebar-upload-zone'),
  docsList:           document.getElementById('docs-list'),
  docsEmpty:          document.getElementById('docs-empty'),
  clearChatBtn:       document.getElementById('clear-chat-btn'),

  // Upload modal
  uploadModal:        document.getElementById('upload-modal'),
  modalDropZone:      document.getElementById('modal-drop-zone'),
  modalCloseBtn:      document.getElementById('modal-close-btn'),
  modalCancelBtn:     document.getElementById('modal-cancel-btn'),
  modalUploadBtn:     document.getElementById('modal-upload-btn'),
  modalFilePreview:   document.getElementById('modal-file-preview'),
  modalFileName:      document.getElementById('modal-file-name'),
  modalFileSize:      document.getElementById('modal-file-size'),
  modalUploadStatus:  document.getElementById('modal-upload-status'),
  pdfInput:           document.getElementById('pdf-input'),

  // Toast
  toast:              document.getElementById('toast'),

  // Suggestion chips
  chips:              document.querySelectorAll('.chip'),
};

/* ============================================================
   APPLICATION STATE
   ============================================================ */
const state = {
  /** Full message history (array of {role, content}) */
  messages: [],

  /** File chosen in the upload modal */
  selectedFile: null,

  /** Whether the AI is currently responding */
  isLoading: false,

  /** Toast auto-hide timer handle */
  toastTimer: null,
};

/* ============================================================
   INITIALISATION
   ============================================================ */
function init() {
  bindEvents();
  checkHealth();
  loadDocuments();
  autoResizeTextarea();
}

/* ============================================================
   EVENT BINDINGS
   ============================================================ */
function bindEvents() {
  // Send on button click
  dom.sendBtn.addEventListener('click', sendMessage);

  // Send on Enter, newline on Shift+Enter
  dom.messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // Auto-resize textarea as user types
  dom.messageInput.addEventListener('input', autoResizeTextarea);

  // Open upload modal from sidebar zone
  dom.sidebarUploadZone.addEventListener('click',  openUploadModal);
  dom.sidebarUploadZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') openUploadModal();
  });

  // Modal controls
  dom.modalCloseBtn.addEventListener('click',  closeUploadModal);
  dom.modalCancelBtn.addEventListener('click', closeUploadModal);

  // Close modal if overlay background clicked
  dom.uploadModal.addEventListener('click', (e) => {
    if (e.target === dom.uploadModal) closeUploadModal();
  });

  // Drop zone in modal → file picker
  dom.modalDropZone.addEventListener('click',  () => dom.pdfInput.click());
  dom.modalDropZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') dom.pdfInput.click();
  });

  // Drag & drop on the modal drop zone
  dom.modalDropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dom.modalDropZone.classList.add('drag-over');
  });
  dom.modalDropZone.addEventListener('dragleave', () => {
    dom.modalDropZone.classList.remove('drag-over');
  });
  dom.modalDropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dom.modalDropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type === 'application/pdf') {
      handleFileSelection(file);
    } else {
      showToast('Please drop a PDF file.', 'error');
    }
  });

  // File input change
  dom.pdfInput.addEventListener('change', () => {
    if (dom.pdfInput.files[0]) {
      handleFileSelection(dom.pdfInput.files[0]);
    }
  });

  // Upload button in modal
  dom.modalUploadBtn.addEventListener('click', uploadPDF);

  // Clear chat button
  dom.clearChatBtn.addEventListener('click', clearChat);

  // Suggestion chips
  dom.chips.forEach((chip) => {
    chip.addEventListener('click', () => {
      const prompt = chip.dataset.prompt;
      if (prompt) {
        dom.messageInput.value = prompt;
        autoResizeTextarea();
        sendMessage();
      }
    });
  });
}

/* ============================================================
   HEALTH CHECK — shows backend online/offline status
   ============================================================ */
async function checkHealth() {
  try {
    const res = await fetch(`${BACKEND_URL}/health`, { method: 'GET' });
    if (res.ok) {
      setStatus('online', 'Backend online');
    } else {
      setStatus('offline', 'Backend error');
    }
  } catch {
    setStatus('offline', 'Backend offline');
  }
}

function setStatus(state, label) {
  dom.statusDot.className = `status-dot ${state}`;
  dom.statusLabel.textContent = label;
}

/* ============================================================
   SEND MESSAGE
   Sends user text to POST /chat and displays AI response.
   ============================================================ */
async function sendMessage() {
  const text = dom.messageInput.value.trim();
  if (!text || state.isLoading) return;

  // Clear input immediately
  dom.messageInput.value = '';
  autoResizeTextarea();

  // Add user message to history and UI
  state.messages.push({ role: 'user', content: text });
  appendMessage('user', text);
  showWelcomeOrMessages();

  // Show thinking indicator, disable send
  showLoading();

  try {
    // NOTE: The backend only reads messages[-1].content (the last message).
    // Sending only the current user message avoids a 422 caused by a typo
    // in the backend schema (role: "assisant" instead of "assistant").
    const response = await fetch(`${BACKEND_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: [{ role: 'user', content: text }],
        temperature: 0.2,
        max_tokens: 512,
      }),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      // FastAPI 422 returns detail as an array of objects, not a plain string.
      // Stringify it properly so the UI never shows "[object Object]".
      const detail = errData.detail;
      const msg = typeof detail === 'string'
        ? detail
        : Array.isArray(detail)
          ? detail.map(e => e.msg || JSON.stringify(e)).join('; ')
          : (detail ? JSON.stringify(detail) : `HTTP ${response.status}`);
      throw new Error(msg);
    }

    const data = await response.json();

    // Extract AI answer from LangGraph response object
    const answer = extractAnswer(data);

    // Store in local history for display only (not sent back to backend)
    state.messages.push({ role: 'assistant', content: answer });
    appendMessage('ai', answer);

  } catch (err) {
    appendError(`Could not reach ForgeMind backend. ${err.message}`);
  } finally {
    hideLoading();
  }
}

/**
 * extractAnswer — parses LangGraph response to get the text answer.
 * LangGraph returns: { answer: "...", question: "..." } or may vary.
 * We try multiple keys as a safety net.
 */
function extractAnswer(data) {
  if (typeof data === 'string') return data;
  if (data.answer)   return data.answer;
  if (data.response) return data.response;
  if (data.content)  return data.content;
  if (data.output)   return data.output;
  if (data.result)   return data.result;
  // Fallback: stringify the whole object so nothing is silently lost
  return JSON.stringify(data, null, 2);
}

/* ============================================================
   UPLOAD PDF
   Sends PDF to POST /upload as multipart form data.
   ============================================================ */
async function uploadPDF() {
  if (!state.selectedFile) {
    showToast('Please choose a PDF first.', 'error');
    return;
  }

  setModalUploadStatus('uploading', '⏳ Uploading...');
  dom.modalUploadBtn.disabled = true;

  const formData = new FormData();
  formData.append('file', state.selectedFile);

  try {
    const response = await fetch(`${BACKEND_URL}/upload`, {
      method: 'POST',
      body: formData,
      // NOTE: Do NOT set Content-Type header when using FormData —
      // the browser sets it automatically with the boundary parameter.
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      const detail = errData.detail;
      const msg = typeof detail === 'string'
        ? detail
        : Array.isArray(detail)
          ? detail.map(e => e.msg || JSON.stringify(e)).join('; ')
          : (detail ? JSON.stringify(detail) : `HTTP ${response.status}`);
      throw new Error(msg);
    }

    const data = await response.json();
    const name  = data.filename || state.selectedFile.name;
    const pages = data.pages   ?? '?';
    const chunks = data.chunks ?? '?';

    setModalUploadStatus('success', `✅ Uploaded! ${pages} pages · ${chunks} chunks indexed.`);
    showToast(`"${name}" uploaded and indexed successfully.`, 'success');

    // Refresh document list
    await loadDocuments();

    // Auto-close after a short pause
    setTimeout(() => {
      closeUploadModal();
    }, 2200);

  } catch (err) {
    setModalUploadStatus('error', `❌ Upload failed: ${err.message}`);
    showToast('Upload failed. Check the backend.', 'error');
    dom.modalUploadBtn.disabled = false;
  }
}

/* ============================================================
   LOAD DOCUMENTS
   GET /documents — lists PDFs that have been uploaded/indexed.
   ============================================================ */
async function loadDocuments() {
  try {
    const res = await fetch(`${BACKEND_URL}/documents/`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    // Accept either {documents: [...]} or a plain array
    const docs = Array.isArray(data) ? data : (data.documents ?? []);
    renderDocumentList(docs);
  } catch {
    // Don't crash — backend may not have documents yet
    renderDocumentList([]);
  }
}

/**
 * renderDocumentList — rebuilds the sidebar document list.
 * @param {Array} docs - array of filename strings or {filename} objects
 */
function renderDocumentList(docs) {
  // Remove all children except the empty placeholder
  const items = dom.docsList.querySelectorAll('.doc-item');
  items.forEach((el) => el.remove());

  if (!docs || docs.length === 0) {
    dom.docsEmpty.style.display = 'block';
    return;
  }

  dom.docsEmpty.style.display = 'none';

  docs.forEach((doc) => {
    const filename = typeof doc === 'string' ? doc : (doc.filename || doc.name || String(doc));
    const li = document.createElement('li');
    li.className = 'doc-item';
    li.innerHTML = `
      <span class="doc-icon">📄</span>
      <span class="doc-name" title="${escapeHtml(filename)}">${escapeHtml(filename)}</span>
      <button class="doc-delete" data-filename="${escapeHtml(filename)}"
              aria-label="Delete ${escapeHtml(filename)}" title="Delete">✕</button>
    `;
    li.querySelector('.doc-delete').addEventListener('click', (e) => {
      e.stopPropagation();
      deleteDocument(filename);
    });
    dom.docsList.appendChild(li);
  });
}

/**
 * deleteDocument — calls DELETE /documents/{filename}
 */
async function deleteDocument(filename) {
  if (!confirm(`Delete "${filename}" from the knowledge base?`)) return;
  try {
    const res = await fetch(`${BACKEND_URL}/documents/${encodeURIComponent(filename)}`, {
      method: 'DELETE',
    });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      const detail = errData.detail;
      const msg = typeof detail === 'string'
        ? detail
        : (detail ? JSON.stringify(detail) : `HTTP ${res.status}`);
      throw new Error(msg);
    }
    showToast(`"${filename}" deleted.`, 'info');
    await loadDocuments();
  } catch (err) {
    showToast(`Could not delete: ${err.message}`, 'error');
  }
}

/* ============================================================
   APPEND MESSAGE — creates and inserts a chat bubble
   ============================================================ */
/**
 * @param {'user'|'ai'} role
 * @param {string} content
 */
function appendMessage(role, content) {
  const isUser = role === 'user';

  const row = document.createElement('div');
  row.className = `message-row ${isUser ? 'user' : 'ai'}`;

  const avatar = document.createElement('div');
  avatar.className = `msg-avatar ${isUser ? 'user-avatar' : 'ai-avatar'}`;
  avatar.setAttribute('aria-hidden', 'true');
  avatar.textContent = isUser ? '👤' : '🧠';

  const bubbleWrapper = document.createElement('div');
  bubbleWrapper.style.display = 'flex';
  bubbleWrapper.style.flexDirection = 'column';
  bubbleWrapper.style.maxWidth = '72%';
  if (isUser) bubbleWrapper.style.alignItems = 'flex-end';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  // Render plain text safely (no innerHTML to avoid XSS)
  bubble.textContent = content;

  const time = document.createElement('div');
  time.className = 'msg-time';
  time.textContent = formatTime(new Date());

  bubbleWrapper.appendChild(bubble);
  bubbleWrapper.appendChild(time);

  if (isUser) {
    row.appendChild(bubbleWrapper);
    row.appendChild(avatar);
  } else {
    row.appendChild(avatar);
    row.appendChild(bubbleWrapper);
  }

  dom.messages.appendChild(row);
  scrollToBottom();
}

/**
 * appendError — shows a full-width error notice in the chat
 */
function appendError(message) {
  const div = document.createElement('div');
  div.className = 'error-bubble';
  div.innerHTML = `⚠️ <span>${escapeHtml(message)}</span>`;
  dom.messages.appendChild(div);
  scrollToBottom();
}

/* ============================================================
   SHOW / HIDE LOADING STATE
   ============================================================ */
function showLoading() {
  state.isLoading = true;
  dom.sendBtn.disabled = true;
  dom.messageInput.disabled = true;
  dom.thinkingRow.classList.add('visible');
  scrollToBottom();
}

function hideLoading() {
  state.isLoading = false;
  dom.sendBtn.disabled = false;
  dom.messageInput.disabled = false;
  dom.thinkingRow.classList.remove('visible');
  dom.messageInput.focus();
}

/* ============================================================
   CLEAR CHAT
   ============================================================ */
function clearChat() {
  if (state.messages.length === 0) return;
  if (!confirm('Clear the entire conversation?')) return;

  state.messages = [];
  dom.messages.innerHTML = '';
  showWelcomeOrMessages();
}

/* ============================================================
   SHOW WELCOME OR MESSAGES depending on chat history
   ============================================================ */
function showWelcomeOrMessages() {
  const hasMessages = state.messages.length > 0;
  dom.welcomeScreen.style.display = hasMessages ? 'none' : 'flex';
  dom.messages.style.display      = hasMessages ? 'flex'  : 'none';
}

/* ============================================================
   UPLOAD MODAL HELPERS
   ============================================================ */
function openUploadModal() {
  state.selectedFile = null;
  dom.pdfInput.value = '';
  dom.modalFilePreview.style.display = 'none';
  dom.modalFileName.textContent = '—';
  dom.modalFileSize.textContent = '—';
  dom.modalUploadBtn.disabled = true;
  hideModalUploadStatus();
  dom.uploadModal.classList.add('visible');
  dom.modalUploadBtn.focus();
}

function closeUploadModal() {
  dom.uploadModal.classList.remove('visible');
}

/**
 * handleFileSelection — called when user picks or drops a PDF
 */
function handleFileSelection(file) {
  state.selectedFile = file;

  dom.modalFileName.textContent = file.name;
  dom.modalFileSize.textContent = formatBytes(file.size);
  dom.modalFilePreview.style.display = 'flex';
  dom.modalUploadBtn.disabled = false;
  hideModalUploadStatus();
}

function setModalUploadStatus(type, message) {
  dom.modalUploadStatus.className = `modal-upload-status visible ${type}`;
  dom.modalUploadStatus.textContent = message;
}

function hideModalUploadStatus() {
  dom.modalUploadStatus.className = 'modal-upload-status';
  dom.modalUploadStatus.textContent = '';
}

/* ============================================================
   TOAST NOTIFICATION
   ============================================================ */
/**
 * @param {string} message
 * @param {'success'|'error'|'info'} type
 * @param {number} duration ms before auto-hide
 */
function showToast(message, type = 'info', duration = 3500) {
  clearTimeout(state.toastTimer);
  dom.toast.textContent = message;
  dom.toast.className = `show ${type}`;
  state.toastTimer = setTimeout(() => {
    dom.toast.className = '';
  }, duration);
}

/* ============================================================
   AUTO-RESIZE TEXTAREA
   Grows vertically as the user types, up to CSS max-height.
   ============================================================ */
function autoResizeTextarea() {
  const el = dom.messageInput;
  el.style.height = 'auto';
  el.style.height = `${el.scrollHeight}px`;
}

/* ============================================================
   UTILITY HELPERS
   ============================================================ */

/** Scroll messages list to the very bottom */
function scrollToBottom() {
  dom.messages.scrollTop = dom.messages.scrollHeight;
}

/** Format a Date as "HH:MM" */
function formatTime(date) {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/** Human-readable byte size */
function formatBytes(bytes) {
  if (bytes < 1024)        return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** Escape HTML special characters to prevent XSS */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

/* ============================================================
   BOOT
   ============================================================ */
document.addEventListener('DOMContentLoaded', init);
