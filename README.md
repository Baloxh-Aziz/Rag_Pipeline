# 📄 RAG Pipeline — PDF Question Answering

A simple **Retrieval-Augmented Generation (RAG)** app built with **Streamlit**. Upload a PDF, let the app process it, then ask questions and get answers based only on the content of your document.

 👉 [Live Demo](https://ragpipeline-assignment.streamlit.app/)

---

## ✨ Features

- Upload any text-based PDF
- Automatically extracts text and splits it into chunks
- Finds the most relevant parts of the document for each question
- Generates an answer using an LLM based on those parts
- Clear warnings for empty questions or PDFs with no extractable text

---

## ⚙️ How It Works

1. **Extract** — text is read from every page of the PDF (`pypdf`).
2. **Chunk** — text is split into chunks of 1000 characters with 200 overlap (`RecursiveCharacterTextSplitter`).
3. **Embed** — each chunk is converted to a vector with `sentence-transformers/all-MiniLM-L6-v2`.
4. **Store** — vectors are stored in a **FAISS** index.
5. **Retrieve** — the 3 most similar chunks to your question are fetched.
6. **Answer** — the question + retrieved chunks are sent to **Google Gemini** (via LangChain) to generate the answer.

---

## 🛠️ Tech Stack

`Python` `Streamlit` `LangChain` `FAISS` `Hugging Face Embeddings` `Google Gemini` `pypdf`

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Baloxh-Aziz/Rag_Pipeline.git
cd Rag_Pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Google API key
mkdir .streamlit
echo 'GOOGLE_API_KEY = "your-api-key-here"' > .streamlit/secrets.toml

# 4. Run the app
streamlit run app.py
```

> 🔐 Never commit your API key. Keep `.streamlit/secrets.toml` out of GitHub (add it to `.gitignore`).

---

## 📁 Project Structure

```
Rag_Pipeline/
├── app.py             # Streamlit app + RAG pipeline
├── requirements.txt   # Python dependencies
└── .devcontainer/     # Dev container config
```

---

## ⚠️ Limitations

- Works with text-based PDFs only (scanned image PDFs need OCR).
- The PDF index is kept in memory, so it resets when the app restarts.

---

## 👤 Author

**Azizullah Asad** 
