# Import packages
import streamlit as st
import pandas as pd
import os
import lida
from sqlalchemy import create_engine
from sqlalchemy.exc import ProgrammingError
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAIEmbeddings
from lida import Manager, TextGenerationConfig, llm
from PIL import Image
from io import BytesIO
import base64
import numpy as np
import random
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from openai import OpenAI as OI
from streamlit_lottie import st_lottie
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from typing_extensions import Concatenate
from Callfuncs import *
from audio_recorder_streamlit import audio_recorder
from streamlit_float import float_init
from dotenv import load_dotenv
from menu import *
from pathlib import Path
load_dotenv()  

BASE_DIR = Path(__file__).resolve().parent
CF2_PATH = BASE_DIR / "cf2.png"
CF3_PATH = BASE_DIR / "cf3.png"

st.set_page_config(
    page_title="Interactive Data to Understand Data",
    page_icon="🍁",
    initial_sidebar_state="collapsed",
    layout="wide",
    menu_items={"Report a bug": "https://docs.google.com/forms/d/1rPpP_jL0r-jWTObnq4JglGjeaNbD8-jjUWslBMgqwPc/edit?pli=1"})

menu()

p0 = st.Page("pagedir/page0.py",title="WELCOME",icon="🍁")
p1 = st.Page("pagedir/page1.py",title="Query existing Data",icon="🤖")
p2 = st.Page("pagedir/page2.py",title="Upload and query Data (CSV / PDF)",icon="📖")
p3 = st.Page("pagedir/page3.py",title="Visualize your Data",icon="📊")
pg = st.navigation({"HOME": [p0],"MODE 1": [p1],"MODE 2": [p2],"MODE 3": [p3]})
# Initialize the Streamlit page


st.title("IRCC 🍁 TICASUK")
st.logo(CF3_PATH, link="https://www.canada.ca/en/immigration-refugees-citizenship.html")
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("cf2.png")
st.html(
    """
<style>
[data-testid="stSidebar"] {
    background-color: black;
}
</style>
"""
)

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{img}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    min-height: 100vh;
}}


.stButton>button {{
        background-color: #26374a;
        color: #FFFFFF;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
    }}
    .stButton>button:hover {{
        background-color: #444444;
    }}
    .stButton>button:active {{
        background-color: #444444;
    }}


[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# selection = st.sidebar.radio("Go to", list(pages.keys()))
# Step 2: Connect to MySQL database
if "db" not in st.session_state:
    st.session_state.db = None

if "connection" not in st.session_state:
    st.session_state.connection = None

# Step 3: Connect button click session
db_uri = "mysql+mysqlconnector://vasant:12345@127.0.0.1/myschema"
st.sidebar.header(body="Configuration",help="Connect and Select LLM")
with st.sidebar:
    connect_button_clicked = st.button("CONNECT")

    if connect_button_clicked:
        try:
            if openai_api_key:
                st.write("OpenAI API key loaded successfully")
            else:
                st.write("Failed to load OpenAI API key")
            st.session_state.db = SQLDatabase.from_uri(db_uri)
            engine = create_engine(db_uri)
            st.session_state.connection = engine.connect()
            st.sidebar.write("Connected to MySQL database")
            st.sidebar.write(st.session_state.db.get_usable_table_names())
        except Exception as e:
            st.sidebar.write(f"An error occurred: {e}")
# Step 4: User selects the LLM model
with st.sidebar:
    model_options = ["gpt-3.5-turbo-0125","gpt-3.5-turbo","gpt-4o-2024-05-13","gpt-4o-mini"]
    selected_model = st.radio("Select Model", model_options, index=model_options.index("gpt-4o-mini"))

    if selected_model == "gpt-4o-mini":
        st.write("Selected gpt-4o-mini")
    elif selected_model == "gpt-3.5-turbo-0125":
        st.write("Selected 3.5-0125 turbo model ")
    elif selected_model == "gpt-3.5-turbo":
        st.write("Select 3.5 turbo model")
    elif selected_model == "gpt-4o-2024-05-13":
        st.write("Selected 4 omni model")
    

    chosen_llm = ChatOpenAI(model=selected_model, temperature=0)
    
# Load agent configuration
if st.session_state.db:
    try:
        db_uri = "mysql+mysqlconnector://vasant:12345@127.0.0.1/myschema"
        db = SQLDatabase.from_uri(db_uri)
        engine = create_engine(db_uri)
        connection = engine.connect()
        st.session_state.db = db
        st.session_state.connection = connection

        # Create the SQL agent
        examples = [
            {"input": "List all unique country names.", "query": "SELECT DISTINCT Country_Name FROM operational;"},
            {
                "input": "Find the total number of permanent resident applications processed for Afghanistan.",
                "query": "SELECT SUM(PR_Permanent_Residents_Applications) FROM operational WHERE Country_Name = 'Afghanistan';",
            },
            {
                "input": "List all records where the number of temporary resident applications processed is greater than 150.",
                "query": "SELECT * FROM operational WHERE TRV_Temporary_Resident_Applications_Processed > 150;",
            },
            {
                "input": "Find the average number of study permits processed per month.",
                "query": "SELECT AVG(SP_Study_Permits_Processed) FROM operational;",
            },
            {
                "input": "List all records for Canada.",
                "query": "SELECT * FROM operational WHERE Country_Name = 'Canada';",
            },
            {
                "input": "How many temporary resident applications were received in total?",
                "query": "SELECT SUM(TRV_Temporary_Resident_Application_Received) FROM operational;",
            },
            {
                "input": "Find the total number of Canadian citizens approved for each country.",
                "query": "SELECT Country_Name, SUM(Canadian_Citizens_approved) FROM operational GROUP BY Country_Name;",
            },
            {
                "input": "List all records where the number of temporary resident applications approved is greater than 100.",
                "query": "SELECT * FROM operational WHERE TRV_Temporary_Resident_Application_Approved > 100;",
            },
            {
                "input": "Who are the top 5 countries by the total number of permanent residents applications?",
                "query": "SELECT Country_Name, SUM(PR_Permanent_Residents_Applications) AS TotalPRApplications FROM operational GROUP BY Country_Name ORDER BY TotalPRApplications DESC LIMIT 5;",
            },
            {
                "input": "Which months have more than 200 authorizations and visas issued for permanent residents?",
                "query": "SELECT Date FROM operational WHERE Authorization_and_Visa_Issued_for_Permanent_Residents > 200;",
            },
            {
                "input": "How many VV1 approvals were there?",
                "query": 'SELECT COUNT(*) FROM operational WHERE VV1_Approved_Count > 0;',
            },
        ]
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples,
            OpenAIEmbeddings(),
            FAISS,
            k=5,
            input_keys=["input"],
        )
        system_prefix = """You are an agent named "Ticasuk" designed to interact with a SQL database. The meaning of your Name is Where the four winds gather their treasures from all parts of the world the greatest of which is knowledge.
        Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
        Always check which table name the user is querying for first.
        Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for the relevant columns given the question.
        You have access to tools for interacting with the database.
        Only use the given tools. Only use the information returned by the tools to construct your final answer.
        You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

        Here are some examples of user inputs and their corresponding MySQL queries:"""
        few_shot_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=PromptTemplate.from_template("User input: {input}\nSQL query: {query}"),
            input_variables=["input", "dialect", "top_k"],
            prefix=system_prefix,
            suffix="",
        )
        full_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate(prompt=few_shot_prompt),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        st.session_state.agent = create_sql_agent(
            llm=chosen_llm,
            db=st.session_state.db,
            prompt=full_prompt,
            verbose=True,
            agent_type="openai-tools",
        )
    except Exception as e:
        st.error(f"Error initializing the agent: {e}")
pg.run()

# Load the selected page
# with open(pages[selection]) as f:
#     page_code = f.read()
# exec(page_code)