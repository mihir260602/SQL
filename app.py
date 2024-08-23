import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq
import pandas as pd
from dotenv import load_dotenv
import os
import time  # Import for loading indicator

# Load environment variables from .env file
load_dotenv()

# Get Groq API key from environment variable
api_key = os.getenv("GROQ_API_KEY")

# Setting up the page configuration with title and icon
st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜",layout="centered")

# Adding custom CSS and HTML for UI enhancement
st.markdown(
    """
    <style>
    /* Background image for the entire app */
    .stApp {
        background-color: #1e1e1e; /* Fallback color if image fails to load */
        color: white;
    }

    /* Custom background for the heading */
    .header-container {
        background-image: url('https://www.icegif.com/space-46/');
        background-size: cover;
        background-position: center;
        padding: 50px;
        border-radius: 15px;
        transition: background 0.5s ease;
    }

    /* Custom font for the title */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@700&display=swap');
    .main-title {
        font-family: 'Roboto', sans-serif;
        font-size: 3em;
        color: white;
        margin-top: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }

    /* Custom font for the tagline */
    .tagline {
        font-family: 'Roboto', sans-serif;
        font-size: 1.5em;
        color: #F0E68C;
        margin-bottom: 30px;
    }

    /* Logo styling */
    .logo {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 150px;
        height: 150px;
    }

    /* Hover effect on the entire container */
    .container:hover {
        transform: scale(1.05);
        transition: transform 0.5s ease;
    }

    /* Make the input box have a transparent background */
    .stTextInput input {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid #F0E68C;
    }

    /* Modify chat input placeholder */
    .stTextInput input::placeholder {
        color: #F0E68C;
    }

    /* Adjust chat message bubbles */
    .stMessage .stMessageContent {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
    }

    /* Loading spinner */
    .spinner {
        border: 4px solid rgba(255, 255, 255, 0.1);
        border-left-color: #F0E68C;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: auto;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    </style>
    """, 
    unsafe_allow_html=True
)

# Add HTML for the heading, tagline, and logo with background image
st.markdown(
    """
    <div class="container header-container">
        <img src="https://th.bing.com/th/id/OIP.Q6FO3FA_rXGiJkF6k6615wAAAA?rs=1&pid=ImgDetMain" alt="Logo" class="logo">
        <h1 class="main-title">LangChain SQL Chatbot</h1>
        <p class="tagline">Empowering your data with AI-driven conversations</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Info messages if API key is not provided
if not api_key:
    st.error("GROQ_API_KEY not found in .env file")

# Initialize the Groq LLM
llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it", streaming=True)

# Function to configure SQLite database
@st.cache_resource(ttl="2h")
def configure_db():
    dbfilepath = (Path(__file__).parent / "analytics_db").absolute()
    creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
    return SQLDatabase(create_engine("sqlite:///", creator=creator))

# Configure DB
db = configure_db()

# SQL toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Creating an agent with SQL DB and Groq LLM
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# Session state for messages (clear button available)
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat history messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input for user query
user_query = st.chat_input(placeholder="Ask anything from the database")

# If user query is submitted
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    # Display loading spinner while processing
    with st.spinner("Processing your query..."):
        time.sleep(1)  # Optional: Add a small delay for UX
        with st.chat_message("assistant"):
            streamlit_callback = StreamlitCallbackHandler(st.container())
            try:
                response = agent.run(user_query, callbacks=[streamlit_callback])
                st.session_state.messages.append({"role": "assistant", "content": response})

                # Handle both final answers and actions
                if isinstance(response, str):
                    # Directly handle string responses
                    st.write(response)
                elif isinstance(response, list):
                    # Handle list responses, assuming they might be in tabular format
                    if all(isinstance(i, tuple) for i in response) and len(response) > 0:
                        # Assuming the first tuple contains the headers
                        headers = [f"Column {i+1}" for i in range(len(response[0]))]
                        df = pd.DataFrame(response, columns=headers)
                        st.dataframe(df.style.set_properties(**{'color': 'white', 'background-color': 'black'}))
                        st.balloons()
                    else:
                        st.write("The response is not in tabular format.")
                else:
                    st.write("Unexpected response format.")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
