# index.py

import fitz
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def build_index_from_pdf(pdf_path, index_path="rag_faiss"): # Renamed function
    """Extracts text from a PDF, chunks it, and builds a FAISS vector index."""
    full_text = extract_text_from_pdf(pdf_path)
    if not full_text:
        print("No text extracted from PDF. Index not built.")
        return None
        
    print(f"Extracted text length: {len(full_text)} characters")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    documents = text_splitter.split_documents([Document(page_content=full_text)])
    print(f"Split text into {len(documents)} documents.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(index_path) if os.path.dirname(index_path) else ".", exist_ok=True)
    vectorstore.save_local(index_path)
    
    print(f"FAISS index with {vectorstore.index.ntotal} vectors saved to {index_path}")
    return vectorstore

if __name__ == "__main__":
    # Ensure you have a PDF file at this path to test the indexing
    pdf_file = "./sample_rental_agreement.pdf"
    if os.path.exists(pdf_file):
        build_index_from_pdf(pdf_file)
        print("\nIndexing complete. You can now use the vector store for similarity search.")
    else:
        print(f"Error: PDF file not found at '{pdf_file}'. Please add it to run the indexing script.")