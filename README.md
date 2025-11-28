📄 Document Q&A Assistant 🤖

A powerful AI-powered Document Question-Answering System built using Streamlit & LangChain


🔍 Overview

Document Q&A Assistant is an AI-powered application that allows users to upload multiple PDF documents and interact with them using natural language questions. It leverages Retrieval-Augmented Generation (RAG) with high-performance LLMs from Groq, FAISS vector search, and HuggingFace embeddings to deliver precise, document-grounded answers.

This tool is ideal for:

Researchers

Students

Analysts

Legal & Financial professionals

Anyone working with large PDFs

✨ Features

📄 Multi-Document Support – Upload and query multiple PDFs at once

💬 Conversational AI – Chat naturally with your documents

🔍 Semantic Search – FAISS-powered similarity search

🎯 Context-Aware Answers – Responses strictly based on document content

💾 Chat Memory – Conversation history persists during sessions

🎨 Modern Dark UI – Clean & productivity-focused interface

⚡ Fast Processing – Optimized chunking and vector indexing

🔐 Secure API Handling – Environment-based key management

🖥️ Tech Stack
Component	Technology
Frontend	Streamlit
Backend	Python
LLM	Groq (Qwen 32B)
Embeddings	all-MiniLM-L6-v2 (HuggingFace)
Vector Database	FAISS
PDF Loader	PyPDFLoader
Chunking	RecursiveCharacterTextSplitter
AI Framework	LangChain
⚙️ System Architecture
PDF Upload → Text Extraction → Chunking → Embeddings → FAISS Index  
User Query → Semantic Search → Relevant Chunks → LLM → Final Answer

🚀 Quick Start
✅ Prerequisites

Python 3.8+

Groq API Key (Get it from Groq Cloud)

(Optional) HuggingFace Token

📦 Installation
git clone https://github.com/your-username/document-qa-assistant.git
cd document-qa-assistant


Create a virtual environment:

python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows


Install dependencies:

pip install -r requirements.txt

🔐 Environment Setup

Create .env file:

cp .env.example .env


Edit .env:

groq_api_key=your_groq_api_key_here
HFToken=your_huggingface_token_here  # Optional

▶️ Run the Application
streamlit run app.py


App will launch at:

http://localhost:8501

📋 How To Use
1️⃣ Configure API Key

Enter your Groq API Key in the sidebar

The AI model initializes automatically

2️⃣ Upload Documents

Click "Upload PDF(s)"

Select one or multiple PDF documents

Files are processed and indexed automatically

3️⃣ Ask Questions

Use chat box to ask questions

Answers are generated only from document content

4️⃣ Manage Conversations

Use Clear Chat & Cache to reset

Upload new PDFs anytime

🛠️ Configuration
🔧 Model Settings
Parameter	Value
LLM	qwen/qwen3-32b
Embeddings	all-MiniLM-L6-v2
Chunk Size	1000 Tokens
Chunk Overlap	50 Tokens
Vector Store	FAISS
📁 Project Structure
document-qa-assistant/
├── app.py                 # Main application logic
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variables template
├── README.md              # Project documentation
└── assets/                # Screenshots & demo files

🚀 Deployment
🔹 Local Deployment
streamlit run app.py

🔹 Cloud Deployment (Optional)

You can deploy on:

Streamlit Cloud

AWS EC2

Render

HuggingFace Spaces

Railway

🔒 Security Best Practices

Never expose your API keys in source code

Always use .env files

Add .env to .gitignore

📈 Performance Tips

Use fewer, high-quality PDFs for faster response

Keep chunk size optimized (1000 works best)

Clear cache when switching documents

🧠 Use Cases

Legal Document Analysis

Academic Research

Financial Reports Q&A

Business Contracts

Technical Manuals

Internal Knowledge Bases

🛠️ Future Enhancements

✅ OCR support for scanned PDFs

✅ Multi-model selection

✅ User authentication

✅ Cloud-based persistent memory

✅ Source-citation highlighting

🤝 Contributing

Contributions are welcome!
Feel free to submit pull requests or open issues for improvements.

📜 License

This project is licensed under the MIT License.

⭐ Support

If you find this project useful, please give it a ⭐ on GitHub — it helps a lot!
