import streamlit as st
from langchain.chains import create\_history\_aware\_retriever, create\_retrieval\_chain
from langchain.chains.combine\_documents import create\_stuff\_documents\_chain
from langchain\_community.vectorstores import Chroma
from langchain\_community.chat\_message\_histories import ChatMessageHistory
from langchain\_core.chat\_history import BaseChatMessageHistory
from langchain\_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain\_groq import ChatGroq
from langchain\_core.runnables.history import RunnableWithMessageHistory
from langchain\_text\_splitters import RecursiveCharacterTextSplitter
from langchain\_community.document\_loaders import PyPDFLoader
from dotenv import load\_dotenv
import os
import tempfile
from langchain\_huggingface import HuggingFaceEmbeddings
import uuid
import sys
import pysqlite3
from PIL import Image

sys.modules\["sqlite3"] = pysqlite3

# Load env variables

load\_dotenv()
os.environ\['HF\_TOKEN'] = st.secrets\["HF\_TOKEN"]
os.environ\["LANGCHAIN\_API\_KEY"] = st.secrets\["LANGCHAIN\_API\_KEY"]
os.environ\["LANGCHAIN\_TRACING\_V2"] = st.secrets\["LANGCHAIN\_TRACING\_V2"]
os.environ\["LANGCHAIN\_PROJECT"] = st.secrets\["LANGCHAIN\_PROJECT"]
api\_key = st.secrets\["GROQ\_API\_KEY"]

# Embeddings

embeddings = HuggingFaceEmbeddings(model\_name="all-MiniLM-L6-v2", model\_kwargs={"device": "cpu"})

# UI Setup

image = Image.open('image.png')
st.set\_page\_config(page\_title="Intelleq", layout="wide")

col1, col2 = st.columns(\[1, 7])
with col1:
st.image(image, width=120)
with col2:
st.markdown("""<h1 >Intelleq <span style='color: #1f77b4; font-size: .7em;  margin-left: 10px;'>- Your AI Assistant</span> </h1>""", unsafe\_allow\_html=True)
st.header(" *Hey, Good to see you here....*")
st.subheader("How can I help you..?")

# Sidebar

st.sidebar.header("🔐 Configuration")
selected\_models = st.sidebar.multiselect(
"Select one or two Open Source models",
\["Gemma2-9b-It", "Deepseek-R1-Distill-Llama-70b", "Qwen-Qwq-32b", "Compound-Beta", "Llama3-70b-8192"],
default=\["Gemma2-9b-It"],
max\_selections=2
)
if not selected\_models:
st.warning("Please select at least one model.")
st.stop()

temperature = st.sidebar.slider("Creativity Level", 0.0, 1.0, 0.7)
language = st.sidebar.selectbox("Select Language", \["English", "Hindi", "Hinglish", "French", "Spanish"], index=0)
st.session\_state.language = language

if not api\_key:
st.warning("Please enter the Groq API Key to continue.")
st.stop()

if "session\_id" not in st.session\_state:
st.session\_state.session\_id = str(uuid.uuid4())
session\_id = st.session\_state.session\_id
st.sidebar.text\_input("Session ID", value=session\_id, disabled=True)
if st.sidebar.button("🔄 New Session"):
st.session\_state.session\_id = str(uuid.uuid4())
st.rerun()

if 'store' not in st.session\_state:
st.session\_state.store = {}
if session\_id not in st.session\_state.store:
st.session\_state.store\[session\_id] = ChatMessageHistory()

# Prompts

contextualize\_q\_prompt = ChatPromptTemplate.from\_messages(\[
("system", "Given a chat history and the latest user question, formulate a standalone question. Do NOT answer it."),
MessagesPlaceholder("chat\_history"),
("human", "{input}"),
])
qa\_prompt = ChatPromptTemplate.from\_messages(\[
("system", "You are a helpful assistant. Use the provided context to answer the question. "
"Always respond to the user **in {language}**, regardless of the input language. "
"If the language is 'Hinglish', respond in Hindi written using English (Roman) script. "
"Be concise, clear, and informative.\n\nContext:\n{context}"),
MessagesPlaceholder("chat\_history"),
("human", "{input}"),
])
evaluation\_prompt = ChatPromptTemplate.from\_messages(\[
("system", "You are an evaluator. Given a question, the assistant's answer, and the context used to generate it, "
"rate the quality of the answer from 1 (poor) to 5 (excellent). Provide a brief justification."),
("human", "Question: {question}\n\nAnswer: {answer}\n\nContext: {context}")
])

# Show Chat History

st.subheader("💬")
chat\_messages = st.session\_state.store.get(session\_id, ChatMessageHistory()).messages\[-20:]
for msg in chat\_messages:
role = "user" if type(msg).**name** == "HumanMessage" else "assistant"
with st.chat\_message(role):
st.markdown(msg.content)

# Evaluator

evaluator = ChatGroq(groq\_api\_key=api\_key, model\_name=selected\_models\[0], temperature=0)

# Evaluate entire conversation

if st.sidebar.button("🔍 Evaluate Entire Conversation"):
session\_history = st.session\_state.store.get(session\_id, ChatMessageHistory()).messages
full\_conversation = ""
for msg in session\_history:
role = "User" if type(msg).**name** == "HumanMessage" else "Assistant"
full\_conversation += f"{role}: {msg.content}\n"

```
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
```

# Chat + File Upload

with st.container():
user\_input = st.chat\_input("Ask a question or upload PDF")
with st.container():
uploaded\_files = st.file\_uploader("📄", type="pdf", accept\_multiple\_files=True, label\_visibility="collapsed")

# Init LLMs

llms = {model: ChatGroq(groq\_api\_key=api\_key, model\_name=model, temperature=temperature) for model in selected\_models}

if user\_input:
session\_history = st.session\_state.store.get(session\_id, ChatMessageHistory())
context\_string = "No PDF uploaded. Use chat history only."

```
# Load and split PDF
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
    context_string = "\n\n".join(doc.page_content for doc in splits[:3])

col1, col2 = st.columns(2) if len(selected_models) == 2 else (st.container(), None)

for i, model_name in enumerate(selected_models):
    model = llms[model_name]
    with (col1 if i == 0 else col2):
        st.markdown(f"### 🤖 Response from `{model_name}`")
        with st.spinner(f"Thinking with {model_name}..."):
            if uploaded_files:
                history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)
                question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
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
            else:
                messages = qa_prompt.format_messages(
                    input=user_input,
                    chat_history=session_history.messages,
                    context=context_string,
                    language=st.session_state.language,
                )
                response = model.invoke(messages)
                assistant_reply = response.content
                session_history.add_user_message(user_input)
                session_history.add_ai_message(assistant_reply)

            st.markdown(assistant_reply)

            eval_messages = evaluation_prompt.format_messages(
                question=user_input,
                answer=assistant_reply,
                context=context_string
            )
            try:
                eval_result = evaluator.invoke(eval_messages)
                with st.expander("🧪 Evaluation Result", expanded=False):
                    st.info(eval_result.content)
            except Exception as e:
                st.warning(f"Evaluation failed: {e}")
