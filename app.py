
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = st.secrets["HF_TOKEN"]


## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]

api_key = st.secrets["GROQ_API_KEY"]

# Initialize HF embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
# Streamlit UI
st.set_page_config(page_title="Conversational PDF Chatbot", layout="wide")
st.title(" Hey, Good to see you here....")
st.subheader("How can i help you..?")

# Sidebar for API key
st.sidebar.header("🔐 Configuration")
model_name=st.sidebar.selectbox("Select Open Source model",["Gemma2-9b-It","Mistral-Saba-24b","Llama3-70b-8192"],index=0)
## Adjust response parameter
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)

if not api_key:
    st.warning("Please enter the Groq API Key to continue.")
    st.stop()

# Initialize LLM
llm = ChatGroq(groq_api_key=api_key, model_name=model_name,temperature=temperature)

# Session ID for chat history
session_id = st.sidebar.text_input("Session ID", value="default_session")
if 'store' not in st.session_state:
    st.session_state.store = {}
if session_id not in st.session_state.store:
    st.session_state.store[session_id] = ChatMessageHistory()

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

# Chat history section at the top
st.subheader("💬")
chat_messages = st.session_state.store.get(session_id, ChatMessageHistory()).messages[-20:]
for msg in chat_messages:
    role = "user" if type(msg).__name__ == "HumanMessage" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Input field with send button inside it (modern chat interface)
# Chat input and file uploader in two rows
with st.container():
    user_input = st.chat_input("Ask a question or upload PDF")
with st.container():
    uploaded_files = st.file_uploader("📄", type="pdf", accept_multiple_files=True, label_visibility="collapsed")
  
# Handle submission
if user_input:
    session_history = st.session_state.store.get(session_id, ChatMessageHistory())
    conversational_rag_chain = None

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temp_pdf = f"./temp_{uploaded_file.name}"
            with open(temp_pdf, "wb") as f:
                f.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temp_pdf)
            documents.extend(loader.load())
            os.remove(temp_pdf)

        # Process docs
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Build RAG chain
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

    with st.spinner("Thinking..."):
        if conversational_rag_chain:
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            assistant_reply = response['answer']
        else:
            response = llm.invoke(user_input)
            assistant_reply = response.content

            # Add to chat history immediately (response will now appear at top)
            session_history.add_user_message(user_input)
            session_history.add_ai_message(assistant_reply)
        st.rerun()
