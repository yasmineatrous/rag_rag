import create_index
import ingest_data  # Ensure the function from ingest_data is imported
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
import retriever
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
from langchain_openai import ChatOpenAI
import nest_asyncio
from dotenv import load_dotenv
import os
from call_llm import *

load_dotenv()
nest_asyncio.apply()

#OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Function to get response from the AI model

def get_response(user_query):
    context = retriever.retrieve_from_pinecone(user_query)[:5]
    print(context)
    st.session_state.context_log = [context]
    
    llm = CustomLLM()
    
    template = """
        You are an AI Syllabus assistant. Answer the question below according to your knowledge in a way that will be helpful to students asking questions about the syllabus.
        The following context is your only source of knowledge to answer from. If you don't know, say you don't know.
        Context: {context}
        User question: {user_question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    return chain.stream({
        "context": context,
        "user_question": user_query
    })

# Streamlit app configuration
st.set_page_config(page_title="Chatbot", page_icon="🤖")
st.title("Spark Chatbot")

# Check if the preprocessing has already been done
if "preprocessing_done" not in st.session_state:
    st.session_state.preprocessing_done = False

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Upload PDF File", type=["pdf"])

if uploaded_file is not None and not st.session_state.preprocessing_done:
    # Display processing message
    st.sidebar.write("Your file is being processed...")
    # Process and upload the PDF to Pinecone
    ingest_data.process_and_upload_pdf(uploaded_file)
    
    # Set the flag to indicate that preprocessing has been done
    st.session_state.preprocessing_done = True
    
    st.sidebar.write("Processing complete. Your data has been uploaded!")

# Initialize chat history and context
if "context_log" not in st.session_state:
    st.session_state.context_log = ["Retrieved context will be displayed here"]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hi, I'm The syllabus assistant. How can I help you?")]

result = st.toggle("Toggle Context")
if result:
    st.write(st.session_state.context_log)

# Display chat messages
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# Input for user queries
user_query = st.chat_input("Type your message here...")
if user_query and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
    
    with st.chat_message("AI"):
        response = st.write_stream(get_response(user_query))
    
    st.session_state.chat_history.append(AIMessage(content=response))
