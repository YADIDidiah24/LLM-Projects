# LexiQE AI - A Legal Document Q&A Assistant

An intelligent AI-powered assistant that allows users to upload legal documents and interact with them through natural language queries. Built with cutting-edge NLP technologies including RAG (Retrieval-Augmented Generation), vector databases, and transformer models.

## 🎯 Features

- 📄 **PDF Document Upload**: Support for various legal document formats
- 🔍 **Intelligent Search**: Semantic search through document content using FAISS vector database
- 💬 **Interactive Chat**: Natural language Q&A interface with conversation history
- 🧠 **Context-Aware Responses**: AI-powered answers based on document content
- ⚡ **Real-Time Processing**: Fast document indexing and query responses
- 🎨 **User-Friendly Interface**: Clean, professional Streamlit web application
- 🔒 **Privacy-First**: All processing happens locally - no data leaves your machine
- 🛠️ **Debug Mode**: View retrieved context for transparency

## 🚀 Demo

![1](https://github.com/user-attachments/assets/1ed0ee3c-6449-4320-80bd-022e7b51179c)


Upload a legal document and start asking questions immediately!

## 🔧 Technology Stack

- **Backend**: Python 3.8+
- **AI/ML**:
  - Hugging Face Transformers (FLAN-T5)
  - LangChain for document processing
  - Sentence Transformers for embeddings
- **Vector Database**: FAISS for similarity search
- **Frontend**: Streamlit
- **Document Processing**: PyMuPDF (fitz)
- **Additional**: PyTorch, NumPy, dotenv

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- At least 4GB of RAM (recommended 8GB+)
- CUDA-compatible GPU (optional, for faster processing)

## 🛠️ Installation

1. **Download the repository**

   ```bash
   git clone https://github.com/yourusername/legal-document-qa.git
   cd legal-document-qa
   ```
2. **Create a virtual environment**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies**
    ```
        pip install -r requirements.txt
    ```



🚀 Usage
Web Application (Recommended)

1. **Start the Streamlit app**
      ``` streamlit run app.py ```
   
3. Access the application

4. Open your browser to http://localhost:8501

5. Upload a legal PDF document

6. Click "Process Document" to build the index

7. Start asking questions!

📁 Project Structure

  ```
  legal-document-qa/
  ├── app.py                 # Main Streamlit application
  ├── index.py              # Document indexing script
  ├── main_chat.py          # CLI chat interface
  ├── tools.py              # Core RAG functionality
  ├── requirements.txt      # Python dependencies
  ├── README.md            # This file
  ├── .env                 # Environment variables (create if needed)
  └── rag_faiss/           # FAISS index storage (auto-created)
  
  ```
## 🔄 How It Works

1. **Document Processing**  
   PDF text is extracted using **PyMuPDF**.

2. **Text Chunking**  
   Documents are split into manageable chunks using **LangChain's RecursiveCharacterTextSplitter**.

3. **Embedding Generation**  
   Text chunks are converted to vector embeddings using **Sentence Transformers**.

4. **Vector Storage**  
   Embeddings are stored in **FAISS** for fast similarity search.

5. **Query Processing**  
   User questions are embedded and matched against stored vectors.

6. **Response Generation**  
   Retrieved context is fed to the **FLAN-T5** model for intelligent answer generation.

---

## 🎯 Example Use Cases

- **Legal Professionals**: Quickly review contracts and agreements  
- **Business Owners**: Understand terms and conditions in legal documents  
- **Students**: Study and analyze legal cases and documents  
- **HR Departments**: Navigate employee handbooks and policies  
- **Real Estate**: Review rental agreements and property contracts  

---

## 📚 Example Questions

- *"What are the payment terms in this contract?"*  
- *"What is the notice period for termination?"*  
- *"Are pets allowed according to this agreement?"*  
- *"What are the maintenance responsibilities?"*  
- *"What penalties apply for late payments?"*

---

## 🙏 Acknowledgments

- **Hugging Face** for providing excellent transformer models  
- **LangChain** for document processing utilities  
- **Streamlit** for the amazing web framework  
- **FAISS** for efficient similarity search  

---

## 🐛 Known Issues

- Large documents (>10MB) may take longer to process  
- Some scanned PDFs may not extract text properly  
- GPU acceleration requires additional CUDA setup  

---

## 🔮 Future Enhancements

- Support for multiple document formats (DOCX, TXT)  
- Multi-language support  
- Document comparison features  
- Export chat history  
- Advanced search filters  
- Integration with legal databases  
- Mobile-responsive design  
- Docker containerization  

---

## 📞 Support

If you encounter any issues or have questions:

- Check the **Issues** page  
- Create a new issue with a detailed description  
- Contact: in Linkedin or in Github

---

## 🌟 Star History

If you find this project helpful, please consider giving it a star! ⭐

---

**Made by Yadidiah K**

> **Disclaimer:** This tool provides AI-generated information and is not a substitute for professional legal advice.

