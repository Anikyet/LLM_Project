import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import SQLiteChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import tempfile
import uuid
import sys
import pysqlite3

# Required for pysqlite3 in some environments
sys.modules["sqlite3"] = pysqlite3

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = st.secrets["HF_TOKEN"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]

api_key = st.secrets["GROQ_API_KEY"]

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# Streamlit UI
st.set_page_config(page_title="Conversational PDF Chatbot", layout="wide")
st.title("Hey, Good to see you here....")
st.subheader("How can I help you..?")

# Sidebar
st.sidebar.header("🔐 Configuration")
model_name = st.sidebar.selectbox("Select Open Source model", ["Gemma2-9b-It", "Mistral-Saba-24b", "Llama3-70b-8192"], index=0)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)

if not api_key:
    st.warning("Please enter the Groq API Key to continue.")
    st.stop()

# Generate or reset session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
session_id = st.session_state.session_id
st.sidebar.text_input("Session ID", value=session_id, disabled=True)
if st.sidebar.button("🔄 New Session"):
    st.session_state.session_id = str(uuid.uuid4())
    st.rerun()

# Prompt templates
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question, "
               "formulate a standalone question. Do NOT answer it."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant. Use the following context to answer concisely.\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Load chat history from SQLite
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return SQLiteChatMessageHistory(session_id=session_id, database_path="chat_history.sqlite")

chat_history = get_session_history(session_id)
st.subheader("💬 Conversation")
for msg in chat_history.messages[-20:]:
    role = "user" if type(msg).__name__ == "HumanMessage" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Input + PDF Upload
with st.container():
    user_input = st.chat_input("Ask a question or upload PDF")
with st.container():
    uploaded_files = st.file_uploader("📄 Upload PDF", type="pdf", accept_multiple_files=True, label_visibility="collapsed")

# Initialize LLM
llm = ChatGroq(groq_api_key=api_key, model_name=model_name, temperature=temperature)

# Process user input
if user_input:
    conversational_rag_chain = None

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                loader = PyPDFLoader(tmp.name)
                documents.extend(loader.load())

        # Split and store documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=None,
            collection_name=f"temp_{session_id}"
        )
        retriever = vectorstore.as_retriever()

        # RAG Chain
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    # Generate answer
    with st.spinner("Thinking..."):
        if conversational_rag_chain:
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            assistant_reply = response['answer']
        else:
            messages = qa_prompt.format_messages(
                input=user_input,
                chat_history=chat_history.messages,
                context="No PDF uploaded. Use chat history only."
            )
            response = llm.invoke(messages)
            assistant_reply = response.content
            chat_history.add_user_message(user_input)
            chat_history.add_ai_message(assistant_reply)

        st.rerun()
