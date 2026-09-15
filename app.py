import gradio as gr
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from pypdf import PdfReader

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0
)

vectorstore = None

def handle_pdf(file):
    global vectorstore
    if file is None:
        return "No PDF uploaded."
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if not text.strip():
            return "Could not extract text from this PDF."

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([text])
        vectorstore = FAISS.from_documents(chunks, embeddings)

        return f"PDF processed! Pages: {len(reader.pages)}, Chunks: {len(chunks)}. Ab sawal pooch sakte ho."
    except Exception as e:
        return f"Error: {str(e)}"
        
def ask_rag(question):
    global vectorstore
    if vectorstore is None:
        return "Pehle PDF upload aur process karo."
    if not question.strip():
        return "Koi sawal likho."
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
        result = qa_chain.invoke({"query": question})
        return result["result"]
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# 📄 Simple RAG PDF Question Answering")
    gr.Markdown("Upload a PDF, process it, and ask questions based on its content.")

    pdf_file = gr.File(label="Upload your PDF", file_types=[".pdf"], type="filepath")
    process_button = gr.Button("Process PDF")
    file_info = gr.Textbox(label="PDF Status", lines=5)
    process_button.click(fn=handle_pdf, inputs=pdf_file, outputs=file_info)

    question = gr.Textbox(label="Ask a question about the PDF", placeholder="e.g. What is the main topic of the document?")
    ask_button = gr.Button("Ask Question")
    answer = gr.Textbox(label="Answer", lines=8)
    ask_button.click(fn=ask_rag, inputs=question, outputs=answer)

demo.launch()
