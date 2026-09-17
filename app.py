import streamlit as st
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from pypdf import PdfReader

st.set_page_config(page_title="Simple RAG — PDF Q&A", page_icon="📄")
st.title("📄 Simple RAG — PDF Question Answering")
st.write("Upload a PDF, process it, then ask questions based on its content.")

# API key comes from Streamlit's secrets, not hardcoded.
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )


@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)


embeddings = load_embeddings()
llm = load_llm()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file and st.button("Process PDF"):
    with st.spinner("Processing PDF..."):
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        if not text.strip():
            st.error("Could not extract text from this PDF.")
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents([text])
            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
            st.success(f"PDF processed! Pages: {len(reader.pages)}, Chunks: {len(chunks)}. You can now ask questions.")

st.divider()

question = st.text_input("Ask a question about the PDF", placeholder="e.g. What is the main topic of the document?")

if st.button("Ask Question"):
    if st.session_state.vectorstore is None:
        st.warning("Please upload and process a PDF first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True
            )
            result = qa_chain.invoke({"query": question})
            st.subheader("Answer")
            st.write(result["result"])
