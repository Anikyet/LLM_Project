import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os
import tempfile
from langchain_huggingface import HuggingFaceEmbeddings
import uuid
import sys
import pysqlite3
from PIL import Image

sys.modules["sqlite3"] = pysqlite3

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = st.secrets["HF_TOKEN"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
api_key = st.secrets["GROQ_API_KEY"]

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# UI Setup
image = Image.open('image.png')
st.set_page_config(page_title="Intelleq", layout="wide")
col1, col2 = st.columns([1, 7])
with col1:
    st.image(image, width=120)
with col2:
    st.markdown("""<h1 >Intelleq
    <span style='color: #1f77b4; font-size: .7em;  margin-left: 10px;'>- Your AI Assistant</span>
    </h1>""", unsafe_allow_html=True)
st.header(" _Hey, Good to see you here...._")
st.subheader("How can I help you..?")

# Sidebar
st.sidebar.header("🔐 Configuration")
model_names = st.sidebar.multiselect(
    "Select up to 2 models for comparison",
    ["Gemma2-9b-It", "Deepseek-R1-Distill-Llama-70b", "Qwen-Qwq-32b", "Compound-Beta", "Llama3-70b-8192"],
    default=["Gemma2-9b-It"], max_selections=2
)
temperature = st.sidebar.slider("Creativity Level", 0.0, 1.0, 0.7)
language = st.sidebar.selectbox("Select Language", ["English", "Hindi", "Hinglish", "French", "Spanish"], index=0)
st.session_state.language = language
if st.sidebar.button("🔄 Refresh"):
    st.rerun()

if not api_key:
    st.warning("Please enter the Groq API Key to continue.")
    st.stop()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
session_id = st.session_state.session_id
st.sidebar.text_input("Session ID", value=session_id, disabled=True)
if st.sidebar.button("🔁 New Session"):
    st.session_state.session_id = str(uuid.uuid4())
    st.rerun()

# Chat History
if 'store' not in st.session_state:
    st.session_state.store = {}
if session_id not in st.session_state.store:
    st.session_state.store[session_id] = ChatMessageHistory()

# Prompt Templates
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question, formulate a standalone question. Do NOT answer it."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Use the provided context to answer the question. "
     "Always respond to the user **in {language}**, regardless of the input language. "
     "If the language is 'Hinglish', respond in Hindi written using English (Roman) script. "
     "Be concise, clear, and informative.\n\nContext:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
evaluation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an evaluator. Given a question, the assistant's answer, and the context used to generate it, "
               "rate the quality of the answer from 1 (poor) to 5 (excellent). Provide a brief justification."),
    ("human", "Question: {question}\n\nAnswer: {answer}\n\nContext: {context}")
])

# Display Chat History
st.subheader("💬")
chat_messages = st.session_state.store.get(session_id, ChatMessageHistory()).messages[-20:]
for msg in chat_messages:
    role = "user" if type(msg).__name__ == "HumanMessage" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

evaluator = ChatGroq(groq_api_key=api_key, model_name=model_names[0], temperature=0)

if st.sidebar.button("🔍 Evaluate Entire Conversation"):
    session_history = st.session_state.store.get(session_id, ChatMessageHistory()).messages
    full_conversation = ""
    for msg in session_history:
        role = "User" if type(msg).__name__ == "HumanMessage" else "Assistant"
        full_conversation += f"{role}: {msg.content}\n"

    eval_messages = evaluation_prompt.format_messages(
        question="Entire conversation",
        answer="Evaluate the entire conversation",
        context=full_conversation
    )
    try:
        eval_result = evaluator.invoke(eval_messages)
    except Exception as e:
        st.error(f"Error occurred during evaluation: {e}")
        st.stop()

    if st.toggle("🔍 Show Evaluation Result", value=True):
        st.subheader("🧪 Full Conversation Evaluation")
        with st.expander("📜 Full conversation context", expanded=False):
            st.text(full_conversation)
        st.info(eval_result.content)

# File Upload
with st.container():
    uploaded_files = st.file_uploader("📄", type="pdf", accept_multiple_files=True, label_visibility="collapsed")

# User Input
with st.container():
    user_input = st.chat_input("Ask a question or upload PDF")

# Display Responses (if available) BEFORE input
if user_input and len(model_names) >= 1:
    st.subheader("🤖 Assistant Responses")
    if len(model_names) == 1:
        st.markdown(f"**Model: `{model_names[0]}`**")
        st.markdown(st.session_state.get("last_answer_0", ""))
    elif len(model_names) == 2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Model: `{model_names[0]}`**")
            st.markdown(st.session_state.get("last_answer_0", ""))
        with col2:
            st.markdown(f"**Model: `{model_names[1]}`**")
            st.markdown(st.session_state.get("last_answer_1", ""))

# Process input
if user_input:
    session_history = st.session_state.store.get(session_id, ChatMessageHistory())
    context_string = "No PDF uploaded. Use chat history only."

    for idx, model_name in enumerate(model_names):
        llm = ChatGroq(groq_api_key=api_key, model_name=model_name, temperature=temperature)
        conversational_rag_chain = None

        if uploaded_files:
            documents = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    temp_pdf = tmp.name
                loader = PyPDFLoader(temp_pdf)
                documents.extend(loader.load())

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=None,
                collection_name=f"temp_{session_id}"
            )
            retriever = vectorstore.as_retriever()
            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            def get_session_history(session: str) -> BaseChatMessageHistory:
                if session not in st.session_state.store:
                    st.session_state.store[session] = ChatMessageHistory()
                return st.session_state.store[session]

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
            response = conversational_rag_chain.invoke(
                {"input": user_input, "language": st.session_state.language},
                config={"configurable": {"session_id": session_id}},
            )
            assistant_reply = response['answer']
            if idx == 0:
                context_string = "\n\n".join(doc.page_content for doc in splits[:3])
        else:
            messages = qa_prompt.format_messages(
                input=user_input,
                chat_history=session_history.messages,
                context=context_string,
                language=st.session_state.language,
            )
            response = llm.invoke(messages)
            assistant_reply = response.content

        # Store each model's output
        st.session_state[f"last_answer_{idx}"] = assistant_reply

        # Save to chat history only for the first model
        if idx == 0:
            session_history.add_user_message(user_input)
            session_history.add_ai_message(assistant_reply)

        # Optional automatic evaluation
        eval_messages = evaluation_prompt.format_messages(
            question=user_input,
            answer=assistant_reply,
            context=context_string
        )
        eval_result = evaluator.invoke(eval_messages)
        if idx == 0:
            st.session_state.last_eval = eval_result.content
            st.session_state.last_question = user_input
            st.session_state.last_answer = assistant_reply
