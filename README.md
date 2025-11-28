<img width="1873" height="780" alt="image" src="https://github.com/user-attachments/assets/851658af-148a-4168-9a4b-309bc874eac8" />

<img width="1863" height="802" alt="image" src="https://github.com/user-attachments/assets/6d00f4ec-3c87-47ba-91bd-9bfa93c5a5cd" />

# 📄 Document Q&A Assistant 🤖

A powerful AI-powered Document Question-Answering System built using Streamlit & LangChain


### *🔍 Overview*

Document Q&A Assistant is an AI-powered application that allows users to upload multiple PDF documents and interact with them using natural language questions. It leverages Retrieval-Augmented Generation (RAG) with high-performance LLMs from Groq, FAISS vector search, and HuggingFace embeddings to deliver precise, document-grounded answers.

This tool is ideal for:

- Researchers

- Students

- Analysts

- Legal & Financial professionals

- Anyone working with large PDFs

## *✨ Features*

1. 📄 Multi-Document Support – Upload and query multiple PDFs at once

2. 💬 Conversational AI – Chat naturally with your documents

3. 🔍 Semantic Search – FAISS-powered similarity search

4. 🎯 Context-Aware Answers – Responses strictly based on document content

5. 💾 Chat Memory – Conversation history persists during sessions

6. 🎨 Modern Dark UI – Clean & productivity-focused interface

7. ⚡ Fast Processing – Optimized chunking and vector indexing

8. 🔐 Secure API Handling – Environment-based key management


## *🚀 Quick Start*
✅ Prerequisites

- Python 3.8+

- Groq API Key (Get it from Groq Cloud)

- (Optional) HuggingFace Token

## *📦 Installation*
- git clone https://github.com/your-username/document-qa-assistant.git-
- cd document-qa-assistant


## Create a virtual environment:
```
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

## Install dependencies:
```
pip install -r requirements.txt
```

## *🔐 Environment Setup*

- Create .env file:

- cp .env.example .env


- Edit .env:
```
groq_api_key=your_groq_api_key_here
HFToken=your_huggingface_token_here  # Optional
```
## ▶️ Run the Application
``` streamlit run app.py ```


### App will launch at:

``` http://localhost:8501 ```

### 📋 How To Use

## 1. Upload Documents

Click "Upload PDF(s)"

Select one or multiple PDF documents

Files are processed and indexed automatically

## 2. Ask Questions

Use chat box to ask questions

Answers are generated only from document content

## 3. Manage Conversations

Use Clear Chat & Cache to reset

Upload new PDFs anytime

## *📁 Project Structure*
document-qa-assistant/
├── app.py                 # Main application logic
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variables template
├── README.md              # Project documentation
└── assets/                # Screenshots & demo files

## 🚀 Deployment
🔹 Local Deployment
streamlit run app.py

# ⭐ Support

If you find this project useful, please give it a ⭐ on GitHub — it helps a lot!
