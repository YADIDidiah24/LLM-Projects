import streamlit as st
import os
import tempfile
import warnings
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up the Streamlit page
st.set_page_config(page_title="Legal Document Q&A", page_icon="⚖️", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #2E86AB;
    margin-bottom: 30px;
}
.upload-section {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.chat-container {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #dee2e6;
}
.question-box {
    background-color: #0a90f2;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}
.answer-box {
    background-color: #5f9e16;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}
.warning-box {
    background-color: #cf1111;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
    border-left: 4px solid #ffc107;
}
.stAlert > div {
    padding: 10px;
}
.stButton>button {
    background-color: #a11d1d;
    color: white;
    border: none;
    padding: 10px 24px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 8px;
}
.stButton>button:hover {
    background-color: #2980b9;
}
.red-button {
    background-color: #d9534f !important;
    color: white !important;
    border: none !important;
    padding: 10px 24px !important;
    text-align: center !important;
    text-decoration: none !important;
    display: inline-block !important;
    font-size: 16px !important;
    margin: 4px 2px !important;
    cursor: pointer !important;
    border-radius: 8px !important;
}

</style>
""", unsafe_allow_html=True)

# Cache the model and tokenizer loading
@st.cache_resource
def load_flan_t5_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        max_length=512
    )

@st.cache_resource
def load_embeddings_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

# Function to process PDF and build vector store
def process_pdf_and_build_vectorstore(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file_path = tmp_file.name
    try:
        text = ""
        with fitz.open(tmp_file_path) as doc:
            for page in doc:
                text += page.get_text()
        if not text.strip():
            st.error("No text could be extracted from the PDF.")
            return None, 0
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=70,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
        docs = text_splitter.split_documents([Document(page_content=text)])
        embeddings = load_embeddings_model()
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore, len(text)
    finally:
        os.unlink(tmp_file_path)

# Function to retrieve legal context
def retrieve_legal_context(query, vectorstore, k=4):
    try:
        docs = vectorstore.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
        return context
    except Exception as e:
        return f"Error retrieving context: {str(e)}"

# Function to generate response using FLAN-T5
def generate_response_with_flan_t5(query, context, pipeline):
    prompt = f"""
You are an expert Legal Assistant. Your task is to answer a user's question based on the provided legal document context.
Your response must be structured, comprehensive, and clear.
Follow these steps carefully:
1. First, directly answer the user's question. Start this section with "Answer:".
2. Second, provide a detailed explanation of the answer based on the context. Elaborate on the key points, definitions, and implications mentioned in the text. Start this section with "Explanation:".
3. Do not add any information that is not present in the context.
4. If the answer cannot be found in the context, state "The provided context does not contain information to answer this question." and do not provide an explanation.
---
Context:
{context}
---
Question:
{query}
---
Response:
"""
    try:
        response = pipeline(
            prompt,
            max_length=512,
            min_length=50,
            num_beams=5,
            early_stopping=True
        )
        return response[0]['generated_text']
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Streamlit UI
st.markdown('<h1 class="main-header">⚖️ Legal Document Q&A Assistant</h1>', unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = load_flan_t5_model()
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

# Sidebar for document upload
with st.sidebar:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    if uploaded_file:
        st.markdown('<style>.stButton>button { background-color: #d9534f; }</style>', unsafe_allow_html=True)
        if st.button("Process Document"):
            with st.spinner("Processing PDF and building index..."):
                st.session_state.vectorstore, doc_length = process_pdf_and_build_vectorstore(uploaded_file)
                if st.session_state.vectorstore:
                    st.success(f"Document processed! ({doc_length:,} chars)")
                    st.session_state.chat_history = []
                else:
                    st.error("Failed to process the document.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.header("2. Settings")
    st.session_state.debug_mode = st.checkbox("Show Retrieved Context (Debug)", value=st.session_state.debug_mode)
    st.markdown('<style>.stButton>button { background-color: #d9534f; }</style>', unsafe_allow_html=True)
    if st.session_state.chat_history and st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Main Chat Interface
if st.session_state.vectorstore is None:
    st.markdown('<div class="warning-box">👋 Welcome! Please upload a legal PDF document to begin.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.header("Ask a Question About Your Document")

    # Quick questions
    quick_questions = [
        "What is this document about?",
        "What are the payment terms?",
        "What are the termination conditions?",
        "Are pets allowed?"
    ]
    cols = st.columns(2)
    for i, q in enumerate(quick_questions):
        if cols[i % 2].button(q):
            st.session_state.user_question = q

    # User input form
    with st.form("question_form"):
        user_question = st.text_input("Your Question:", key="user_question", placeholder="e.g., What is the notice period for termination?")
        submit_button = st.form_submit_button("Get Answer")

    if submit_button and user_question:
        with st.spinner("Searching document and generating answer..."):
            context = retrieve_legal_context(user_question, st.session_state.vectorstore)
            answer = generate_response_with_flan_t5(user_question, context, st.session_state.pipeline)
            st.session_state.chat_history.insert(0, {"question": user_question, "answer": answer, "context": context})
            st.rerun()

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        for chat in st.session_state.chat_history:
            st.markdown(f'<div class="question-box">❓ You: {chat["question"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-box">💡 Assistant: {chat["answer"]}</div>', unsafe_allow_html=True)
            if st.session_state.debug_mode:
                with st.expander("Show Retrieved Context"):
                    st.text(chat["context"])
    st.markdown('</div>', unsafe_allow_html=True)

# Disclaimer Footer
st.markdown("---")
st.markdown('<p style="text-align: center; color: grey;">⚠️ Disclaimer: This tool provides AI-generated information and is not a substitute for professional legal advice.</p>', unsafe_allow_html=True)
