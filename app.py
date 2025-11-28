import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from typing import List, Dict, Any
import tempfile

# Load environment variables
load_dotenv()

# ---------------------------
# Minimal, non-invasive CSS
# Goal: avoid large white "cards" / big white boxes while keeping a modern look
# ---------------------------
def load_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        :root {
            --bg:#0f1724;        /* page background (dark) */
            --panel:#0b1220;     /* panel background (very dark) */
            --muted:#94a3b8;
            --accent:#3b82f6;
            --card-alpha: rgba(255,255,255,0.02);
        }

        html, body, [data-testid="stAppViewContainer"] > div:first-child {
            background: var(--bg) !important;
            color: #e6eef8;
            font-family: 'Inter', sans-serif;
        }

        /* Make the main Streamlit container blend with the page (no big white box) */
        .css-1lcbmhc.e1fqkh3o2, /* top-level block container */
        .block-container {
            background: transparent !important;
            padding-top: 8px;
            padding-left: 16px;
            padding-right: 16px;
        }

        /* Tighter spacing and subtle panels instead of white cards */
        .panel {
            background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
            border: 1px solid rgba(255,255,255,0.03);
            padding: 14px;
            border-radius: 10px;
            margin-bottom: 12px;
        }

        /* Streamlit elements styling (inputs, buttons) with subtle dark theme */
        .stTextInput input, .stTextArea textarea {
            background: transparent !important;
            color: #e6eef8 !important;
            border: 1px solid rgba(255,255,255,0.06) !important;
            border-radius: 8px;
            padding: 8px;
        }

        .stButton button {
            background: linear-gradient(180deg, var(--accent), #2563EB) !important;
            color: white !important;
            border: none !important;
            padding: 8px 14px !important;
            border-radius: 8px !important;
            box-shadow: 0 6px 18px rgba(59,130,246,0.12) !important;
        }

        /* Chat bubble look but small and compact */
        .user-message {
            background: linear-gradient(90deg, rgba(59,130,246,0.12), rgba(59,130,246,0.06));
            color: #e6eef8;
            border-radius: 14px;
            padding: 8px 12px;
            margin: 6px 0;
            max-width: 78%;
            margin-left: auto;
            font-size: 14px;
        }
        .assistant-message {
            background: rgba(255,255,255,0.02);
            color: #cbd5e1;
            border-radius: 14px;
            padding: 8px 12px;
            margin: 6px 0;
            max-width: 78%;
            margin-right: auto;
            font-size: 14px;
        }

        /* File uploader subtle frame */
        .uploadedFile {
            border: 1px dashed rgba(255,255,255,0.04);
            background: rgba(255,255,255,0.01);
            padding: 10px;
            border-radius: 8px;
        }

        /* Hide Streamlit menu/footer to keep UI compact */
        #MainMenu, footer, header {visibility: hidden;}

        /* Responsive small typography for compactness */
        h1, h2, h3, h4 { margin: 6px 0; color: #e6eef8; }
        p, li { color: #cbd5e1; }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------
# Session state initialisation
# ---------------------------
def initialize_session_state() -> None:
    if "store" not in st.session_state:
        st.session_state.store = {}
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "api_key_verified" not in st.session_state:
        st.session_state.api_key_verified = False


# ---------------------------
# Cached model initialisation
# ---------------------------
@st.cache_resource(show_spinner=False)
def initialize_model(api_key: str) -> ChatGroq:
    try:
        return ChatGroq(model="qwen/qwen3-32b", api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize model: {e}")
        return None


# ---------------------------
# Document processing (cached)
# ---------------------------
@st.cache_data(show_spinner=False)
def process_documents(uploaded_files: List) -> tuple:
    try:
        documents = []
        temp_files = []

        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                tmp.flush()
                temp_files.append(tmp.name)
                loader = PyPDFLoader(tmp.name)
                docs = loader.load()
                documents.extend(docs)

        # Split documents into compact chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        splitted = splitter.split_documents(documents)

        # Create embeddings and vectorstore
        embeddings = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splitted, embeddings)

        # cleanup
        for path in temp_files:
            try:
                os.unlink(path)
            except Exception:
                pass

        return vectorstore, len(splitted)

    except Exception as e:
        st.error(f"Error processing documents: {e}")
        return None, 0


# ---------------------------
# Sidebar: compact controls
# ---------------------------
def render_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        api_key = os.getenv("groq_api_key")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### 📁 Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF(s)",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDFs"
        )
        if uploaded_files:
            st.write(f"Uploaded: {len(uploaded_files)}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Actions")
        if st.button("🗑 Clear chat & cache", use_container_width=True):
            st.session_state.messages = []
            st.session_state.store = {}
            st.session_state.vectorstore = None
            st.session_state.documents_processed = False
            st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    return {"api_key": api_key, "uploaded_files": uploaded_files}


# ---------------------------
# Chat rendering (compact, no large white boxes)
# ---------------------------
def render_chat_interface(_model: ChatGroq, vectorstore: FAISS) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### 💬 Document Q&A")

    # compact message display (use small chat bubbles)
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{st.session_state.get("escape_html", False) and st.utils.safely_format_html(message["content"]) or message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">{st.session_state.get("escape_html", False) and st.utils.safely_format_html(message["content"]) or message["content"]}</div>', unsafe_allow_html=True)

    # chat input (compact)
    prompt = st.chat_input("Ask something about the uploaded documents...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Show user bubble immediately
        st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)

        # Generate response
        with st.spinner("Generating answer..."):
            try:
                retriever = vectorstore.as_retriever()

                prompt_rewritten_query = ChatPromptTemplate.from_messages([
                    ("system", "You are an AI assistant that rewrites the user's query based on the chat history. You must output a fully rewritten standalone query.If the user's question already makes sense independently, return it unchanged."),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ])

                history_aware_retriever = create_history_aware_retriever(_model, retriever, prompt_rewritten_query)

                prompt_for_model = ChatPromptTemplate.from_messages([
                            (
                                "system",
                                "You are an AI assistant that answers the user's questions strictly using the information "
                                "from the provided documents. You also consider the conversation history to maintain "
                                "continuity. Follow these rules:\n\n"
                                
                                "1. Only use facts present in the retrieved documents.\n"
                                "2. If the answer is not found inside the documents, say: "
                                "'The uploaded documents do not contain information about this.'\n"
                                "3. Never hallucinate or make up facts.\n"
                                "4. If the user asks something unrelated to the documents, politely decline.\n"
                                "5. Provide clear, step-by-step, correct, and helpful answers.\n"
                                "6. Do NOT reveal chain-of-thought or reasoning. \n"
                                "7. Only provide the final answer in a clear and concise way. \n"
                                "7. Very Important: Make sure not to provide think content in answer just the output only nothing else.  Never include phrases like <think> … </think>.\n"
                                "Here are the retrieved and relevant document chunks:\n\n{context}\n\n"
                            ),
                            ("human", "{input}")
                        ])

                model_with_context = create_stuff_documents_chain(_model, prompt_for_model)
                rag_chain = create_retrieval_chain(history_aware_retriever, model_with_context)

                def with_session_chat(session_id) -> BaseChatMessageHistory:
                    if session_id not in st.session_state.store:
                        st.session_state.store[session_id] = ChatMessageHistory()
                    return st.session_state.store[session_id]

                session_rag_chain = RunnableWithMessageHistory(
                    rag_chain,
                    with_session_chat,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer",
                )

                configuration = {"configurable": {"session_id": "chat1"}}
                response = session_rag_chain.invoke({"input": prompt}, config=configuration)

                if response and "answer" in response:
                    assistant_answer = response["answer"]
                    st.markdown(f'<div class="assistant-message">{assistant_answer}</div>', unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": assistant_answer})
                else:
                    st.error("No response generated from the model.")

            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    st.set_page_config(page_title="Document Q&A Assistant", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
    load_css()
    initialize_session_state()

    # Header (compact)
    left, right = st.columns([3, 1])
    with left:
        st.markdown("<h1>📚 Document Q&A Assistant</h1>", unsafe_allow_html=True)
        st.markdown("<p class='muted'>Upload PDFs and ask factual questions about their contents.</p>", unsafe_allow_html=True)
    with right:
        if st.session_state.documents_processed:
            st.success("✅ Ready")
        else:
            st.info("Upload PDFs to start")

    st.markdown("---")

    # Sidebar controls
    config = render_sidebar()

    # If API key present, initialize model
    if config["api_key"]:
        model = initialize_model(config["api_key"])
        if not model:
            st.error("Invalid API key or model init failed.")
            return

        # If uploads present, process them (compact UI)
        if config["uploaded_files"]:
            left_col, right_col = st.columns([3, 1])
            with left_col:
                if not st.session_state.documents_processed or st.session_state.vectorstore is None:
                    with st.spinner("Processing documents..."):
                        vectorstore, chunk_count = process_documents(config["uploaded_files"])
                        if vectorstore:
                            st.session_state.vectorstore = vectorstore
                            st.session_state.documents_processed = True
                            st.session_state.chunk_count = chunk_count
                            # don't force rerun; continue execution
                if st.session_state.documents_processed and st.session_state.vectorstore:
                    render_chat_interface(model, st.session_state.vectorstore)
            with right_col:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown("### ℹ️ Quick tips")
                st.markdown("- Ask specific questions for accurate answers.")
                st.markdown("- The assistant only uses uploaded documents.")
                st.markdown(f"- Text chunks: {st.session_state.get('chunk_count', '—')}")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            # No documents uploaded yet: short getting-started panel
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("### Get started")
            st.markdown("1. Upload PDF files\n2. Ask questions about the documents")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        # API key missing: guide user briefly
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### API Key required")
        st.markdown("Please provide your Groq API key in the sidebar to initialize the model.")
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
