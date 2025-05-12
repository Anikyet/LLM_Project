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
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from PIL import Image
import tempfile
import os
import uuid
import sys
import pysqlite3
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Monkey patch sqlite3 for Chroma
sys.modules["sqlite3"] = pysqlite3

# Setup
load_dotenv()
nltk.download("punkt")

os.environ['HF_TOKEN'] = st.secrets["HF_TOKEN"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
api_key = st.secrets["GROQ_API_KEY"]

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

# UI
image = Image.open('image.png')
st.set_page_config(page_title="Intelleq", layout="wide")

col1, col2 = st.columns([1, 7])
with col1:
    st.image(image, width=120)
with col2:
    st.markdown("""<h1>Intelleq<span style='color: #1f77b4; font-size: .7em; margin-left: 10px;'> - Your AI Assistant</span></h1>""", unsafe_allow_html=True)

st.header(" _Hey, Good to see you here...._")
st.subheader("How can I help you..?")

# Sidebar
st.sidebar.header("🔐 Configuration")
model_name = st.sidebar.selectbox("Select Open Source model", ["Gemma2-9b-It", "Deepseek-R1-Distill-Llama-70b", "Qwen-Qwq-32b", "Compound-Beta", "Llama3-70b-8192"], index=0)
temperature = st.sidebar.slider("Creativity Level", 0.0, 1.0, 0.7)
language = st.sidebar.selectbox("Select Language", ["English", "Hindi", "Hinglish", "French", "Spanish"], index=0)
st.session_state.language = language

if not api_key:
    st.warning("Please enter the Groq API Key to continue.")
    st.stop()

# Session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
session_id = st.session_state.session_id
st.sidebar.text_input("Session ID", value=session_id, disabled=True)
if st.sidebar.button("🔄 New Session"):
    st.session_state.session_id = str(uuid.uuid4())
    st.rerun()

if 'store' not in st.session_state:
    st.session_state.store = {}
if session_id not in st.session_state.store:
    st.session_state.store[session_id] = ChatMessageHistory()

# Prompts
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question, formulate a standalone question. Do NOT answer it."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the provided context to answer the question. Always respond in {language}.\n\nContext:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Evaluation Function
def evaluate_answer(reference, generated):
    ref_tokens = nltk.word_tokenize(reference)
    gen_tokens = nltk.word_tokenize(generated)

    # BLEU
    smoothie = SmoothingFunction().method4
    bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothie)

    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)

    return {
        "BLEU": round(bleu, 3),
        "ROUGE-1": round(scores["rouge1"].fmeasure, 3),
        "ROUGE-L": round(scores["rougeL"].fmeasure, 3),
    }

# Show Chat
st.subheader("💬")
chat_messages = st.session_state.store[session_id].messages[-20:]
for msg in chat_messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Chat input and file upload
user_input = st.chat_input("Ask a question or upload PDF")
uploaded_files = st.file_uploader("📄", type="pdf", accept_multiple_files=True, label_visibility="collapsed")

# LLM
llm = ChatGroq(groq_api_key=api_key, model_name=model_name, temperature=temperature)

if user_input:
    session_history = st.session_state.store[session_id]
    conversational_rag_chain = None
    context_string = "No PDF uploaded. Using chat history only."

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
        vectorstore = Chroma.from_documents(splits, embedding=embeddings, persist_directory=None, collection_name=f"temp_{session_id}")
        retriever = vectorstore.as_retriever()

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            return st.session_state.store.setdefault(session, ChatMessageHistory())

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    with st.spinner("Thinking..."):
        if conversational_rag_chain:
            response = conversational_rag_chain.invoke(
                {"input": user_input, "language": language},
                config={"configurable": {"session_id": session_id}},
            )
            assistant_reply = response['answer']
            context_string = "\n\n".join(doc.page_content for doc in splits[:3])
        else:
            messages = qa_prompt.format_messages(
                input=user_input,
                chat_history=session_history.messages,
                context=context_string,
                language=language,
            )
            response = llm.invoke(messages)
            assistant_reply = response.content

        # Display response
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)

        # Evaluation
        eval_scores = evaluate_answer(user_input, assistant_reply)
        st.session_state.last_eval = eval_scores
        st.session_state.last_question = user_input
        st.session_state.last_answer = assistant_reply

        # Show evaluation
        with st.expander("🧪 Evaluation (BLEU & ROUGE)"):
            st.write(eval_scores)

        # Save chat
        if not conversational_rag_chain:
            session_history.add_user_message(user_input)
            session_history.add_ai_message(assistant_reply)

        st.rerun()
