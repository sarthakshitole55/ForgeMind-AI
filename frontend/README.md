# ForgeMind AI — Frontend

> **Static HTML/CSS/Vanilla JS frontend for the ForgeMind AI Industrial Assistant**

No build tools. No npm. No frameworks. Just open `index.html` in a browser.

---

## 📁 Folder Structure

```
frontend/
│
├── index.html     # Main HTML page — layout and all DOM elements
├── style.css      # All styles — dark theme, flexbox, responsive
├── script.js      # All JavaScript — fetch, chat, upload, UI logic
├── assets/        # Static assets (icons, images — currently empty)
└── README.md      # This file
```

---

## 🚀 How to Run

### Option 1 — Open directly (simplest)

```bash
# From the project root
open frontend/index.html       # macOS
xdg-open frontend/index.html   # Linux
```

> ⚠️ Some browsers block `fetch()` cross-origin requests when opening a file directly.
> Use Option 2 (recommended) to avoid CORS issues.

### Option 2 — Python simple HTTP server (recommended)

```bash
cd frontend
python3 -m http.server 3000
```

Then open: [http://localhost:3000](http://localhost:3000)

### Option 3 — VS Code Live Server

Install the **Live Server** extension and right-click `index.html` → *Open with Live Server*.

---

## 🔗 Backend URL

The frontend talks to the FastAPI backend at:

```
http://localhost:8000
```

If your backend runs on a different port or host, change the constant at the **top of `script.js`**:

```js
const BACKEND_URL = 'http://localhost:8000';
```

---

## 🌐 Backend API Communication

| Action | Method | Endpoint | Payload |
|---|---|---|---|
| Health check | `GET` | `/` | — |
| Send a message | `POST` | `/chat` | `{ messages: [{role, content}], temperature, max_tokens }` |
| Upload PDF | `POST` | `/upload` | `FormData` with `file` field |
| List documents | `GET` | `/documents/` | — |
| Delete document | `DELETE` | `/documents/{filename}` | — |

All communication uses `fetch()` with `async/await`. No Axios, no external HTTP libraries.

---

## 🔧 CORS Configuration

The backend currently allows only `http://localhost:5173`.

To use this frontend served from port 3000 (or directly as a file), add your origin to `backend/app/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",   # ← add this
        "http://127.0.0.1:3000",   # ← and this
        "null",                    # ← for file:// access (dev only)
    ],
    ...
)
```

---

## 💡 Key JavaScript Functions

| Function | What it does |
|---|---|
| `sendMessage()` | Reads textarea, calls `POST /chat`, renders AI reply |
| `uploadPDF()` | Sends selected file to `POST /upload` via FormData |
| `appendMessage(role, content)` | Creates a chat bubble and appends it to the DOM |
| `showLoading()` | Shows "ForgeMind is thinking…" and disables the send button |
| `hideLoading()` | Hides the thinking indicator and re-enables input |
| `loadDocuments()` | Fetches `GET /documents/` and renders the sidebar list |
| `showToast(msg, type)` | Shows a small notification (success / error / info) |
| `clearChat()` | Wipes the conversation history and resets the UI |

---

## 🎨 Design Decisions

- **Dark theme** with CSS custom properties (`--bg-primary`, `--accent-blue`, etc.)
- **Flexbox** used for all layouts — no CSS Grid, no floats
- **No animations** except a subtle `.btn:hover` scale and thinking dots bounce
- **Responsive** — sidebar hides on screens narrower than 640 px
- **Accessible** — `aria-label`, `role="log"`, `aria-live` on chat for screen readers
- **XSS-safe** — all user-provided text goes through `escapeHtml()` or `.textContent`

---

## 📝 Files Explained

### `index.html`
The only HTML page. Contains the full layout:
- `<header>` with logo, title, backend status indicator
- `<aside>` sidebar with upload zone + document list
- `<main>` chat area with welcome screen, messages, thinking indicator, and input bar
- Upload modal (hidden by default, shown on click)
- Toast notification element
- Loads `style.css` and `script.js` — nothing else

### `style.css`
All visual styling in one file:
- CSS custom properties at `:root` level for easy theming
- Sections clearly commented: header, sidebar, chat, input, modal, toast
- Zero JavaScript-in-CSS (no Tailwind, no CSS-in-JS)

### `script.js`
All frontend logic in one file:
- `const BACKEND_URL` at the top — single place to change the API host
- `const dom = {...}` — all DOM element references cached once at startup
- `const state = {...}` — all mutable state in one object
- Functions are small, focused, and named for what they do
- Every `fetch()` call is wrapped in `try/catch` with user-friendly error display

### `assets/`
Empty directory reserved for future icons, images, or logos.

---

## 🛡️ Security Notes

- Never runs `eval()` or `innerHTML` with user content
- All backend responses rendered via `.textContent` (not `.innerHTML`)
- `escapeHtml()` used wherever text goes into `innerHTML` (e.g., filenames)
