# InquizoAI

**Summarize Smarter. Question Deeper.**

A Flask web app for summarizing text, PDFs, and YouTube videos, and for answering questions from your content or the web.

---

## Features

- Summarize text, PDF, or YouTube video content using state-of-the-art AI.
- **Combine multiple sources:** If you provide any combination of text, PDF, and YouTube link, the app will extract and mix all content before summarizing or answering questions.
- **Smart question answering:** For definition-type or generic questions (e.g., 'What is...', 'Define...', 'Explain...'), the system uses a generative LLM to provide a more informative answer, ensuring high-quality responses even when the content is vague or missing details.
- Ask questions about your content and get answers from the content, the web, or an LLM.
- Modern, responsive web UI (Bootstrap).
- Handles large files with user warnings for free hosting limits.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/inquizoai.git
cd inquizoai
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set environment variables (for Render or locally)

- `GOOGLE_API_KEY`
- `GOOGLE_CSE_ID`
- `SERPAPI_KEY`
- (Optional) `HUGGINGFACE_TOKEN`

### 5. Run the app

```bash
python app.py
```

Visit [http://localhost:5000](http://localhost:5000) in your browser.

---

## Deployment

- Ready for deployment on [Render](https://render.com/) (see `Procfile`).
- Health check endpoint: `/healthz`
- Uses `PORT` environment variable for compatibility.

---

## Notes

- Free hosting (like Render) has file size and memory limits. Large PDF uploads may fail.
- All processing is ephemeral; no persistent storage is used.

---

## License

MIT 