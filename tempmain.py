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
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from typing_extensions import Concatenate
from testingDirectory.voice_func import *
from audio_recorder_streamlit import audio_recorder
from streamlit_float import float_init
from dotenv import load_dotenv

# MySQL connection string
db_uri = "mysql+mysqlconnector://vasant:12345@127.0.0.1/myschema"
# Streamlit main configuration
st.set_page_config("IRCC Data Analysis")
st.title("IRCC Data Analysis")

#OPENAI KEY CONFIGURATION
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)
float_init()
footer_container = st.container()

# Step 1: User selects the operation mode
operation_mode = st.selectbox(
    "Select Mode of Operation",
    ["Query Existing Data", "Upload and Query New Data", "Visualize Your Data"]
)

# Step 2: Connect to MySQL database
if "db" not in st.session_state:
    st.session_state.db = None

if "connection" not in st.session_state:
    st.session_state.connection = None

# Step 3: Connect button click session
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
    selected_model = st.radio("Select Model", model_options)
    if selected_model == "gpt-3.5-turbo-0125":
        st.write("Selected 3.5-0125 turbo model ")
    elif selected_model == "gpt-3.5-turbo":
        st.write("Select 3.5 turbo model")
    elif selected_model == "gpt-4o-2024-05-13":
        st.write("Selected 4 omni model")
    elif selected_model == "gpt-4o-mini":
        st.write("Selected gpt-4o-mini")

    chosen_llm = ChatOpenAI(model=selected_model, temperature=0)

# Initialize and train the SQL AGENT
# Feed Examples to the model
if st.session_state.db:
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
    # Convert the Example SQL Queries to Embeddings FAISS
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(),
        FAISS,
        k=5,
        input_keys=["input"],
    )
    # Training the model
    system_prefix = """You are an agent named "Ticasuk" designed to interact with a SQL database. The meaning of your Name is Where the four winds gather their treasures from all parts of the world the greatest of which is knowledge.
    Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
    Always check which table name the user is querying for first.
    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.
    Only use the given tools. Only use the information returned by the tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database. If user asks to make DML statements return "Not allowed to Make DML statements".

    Return "The answer is 0 but is imputed" where the response to a user's query is only INT "0".

    Here are some examples of user inputs and their corresponding MySQL queries:"""
    # Few-shot Learning involves training a model with a very limited number of examples per class, 
    # enabling it to generalize and make accurate predictions with minimal data. 
    # This approach is useful in scenarios where acquiring a large dataset is challenging or expensive.
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate.from_template(
            "User input: {input}\nSQL query: {query}"
        ),
        input_variables=["input", "dialect", "top_k"],
        prefix=system_prefix,
        suffix="",
    )
    # Full-shot learning uses a large and comprehensive dataset with many examples per class 
    # leading to higher accuracy and robustness due to the extensive amount of training data.
    # full-shot learning benefits from the abundance of data to improve model performance.
    full_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate(prompt=few_shot_prompt),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    # Creation of sqlagent based on training
    agent = create_sql_agent(
        llm=chosen_llm,
        db=st.session_state.db,
        prompt=full_prompt,
        verbose=True,
        agent_type="openai-tools",
    )

    # Final: SQL agent responds to the queries based on the selected mode
    if operation_mode == "Query Existing Data":
        with footer_container:
            recorded_audio = audio_recorder(text="Voice Analysis: Click to Record",recording_color="#FF7F7F",neutral_color="#7B0323")
        user_input = st.text_input("Enter your SQL query")
        query_button = st.button("ASK YOUR QUERY")
        if user_input is not None and query_button:
                with st.spinner("Analyzing..."):
                    try:
                        sqlagentresponse = agent.invoke({"input": user_input})
                    except Exception as e:
                        st.write(f"Failed to process the query: {e}")
                    finally:
                        st.write(sqlagentresponse["output"])
        elif recorded_audio:
            try:
                audio_file = "audio.mp3"
                with open(audio_file,"wb") as f:
                    f.write(recorded_audio)
                transcribed_text = speech_to_text(audio_file)
                agentTTS = agent.invoke({"input": transcribed_text})
                str_final_output = agentTTS["output"]
                response_audio_file = "audio_response.mp3"
                text_to_audio(str_final_output,response_audio_file)
                st.audio(response_audio_file)
                autoplay_audio((response_audio_file))
            except Exception as e:
                st.error(f"No Voice Detected, Retry please❗ ")
            # st.write("Your Query: ", transcribed_text)
            finally:
                final_output = st.write(agentTTS["output"])
                
    elif operation_mode == "Upload and Query New Data":
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx","pdf"])
        if uploaded_file is not None:
            # WHEN FILE FORMAT IS CSV XLSX OR XLS
            if uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
                file_name = uploaded_file.name
                base_name = os.path.splitext(file_name)[0]

                if file_name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif file_name.endswith(('.xlsx','xls')):
                    df = pd.read_excel(uploaded_file)
                        
                st.write(f"The filename is: {base_name}")
                st.write(df)
                
                df.columns = df.columns.str.strip()

                if st.session_state.connection is not None:
                    try:
                        df.to_sql(base_name, st.session_state.connection, if_exists='fail', index=False)
                        st.write(f"Data has been successfully loaded into the table `{base_name}`")
                    except Exception as e:
                        if "Table '{}' already exists".format(base_name) in str(e):
                            st.write(f"The table `{base_name}` already exists in the database.")
                        else:
                            st.write(f"An error occurred while inserting data into MySQL: {e}")
                else:
                    st.write("Database connection is not established.")

                user_input = st.text_input("Enter your SQL query for the new data")
                if st.button("ASK YOUR QUERY FOR NEW DATA"):
                    with st.spinner("Analyzing..."):
                        try:
                            sqlagentresponse = agent.invoke({"input": user_input})
                            st.write(sqlagentresponse["output"])
                        except Exception as e:
                            st.write(f"Failed to process the query: {e}")
            # WHEN FILE IS PDF FORMAT
            if uploaded_file.name.endswith('.pdf'):
                embeddings = OpenAIEmbeddings(api_key=openai_api_key)
                pdf_reader = PdfReader(uploaded_file)
                num_pages = len(pdf_reader.pages)
                file_name = uploaded_file.name
                base_name = os.path.splitext(file_name)[0]
                st.write(f"The filename is: {base_name} and it has {num_pages} pages")
                
                def set_random_seed(seed):
                    np.random.seed(seed)
                    random.seed(seed)
                
                set_random_seed(60)   
                
                raw_text = ''
                for i, page in enumerate(pdf_reader.pages):
                    content = page.extract_text()
                    if content:
                        raw_text += content
            
                text_splitter = CharacterTextSplitter(
                    separator = "\n",
                    chunk_size = 1000,
                    chunk_overlap  = 200,
                    length_function = len
                )
                texts = text_splitter.split_text(raw_text)
    
                document_search = FAISS.from_texts(texts, embeddings)
                query = st.text_input("Enter your Query for the pdf")
                docs = document_search.similarity_search(query)
                chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

                if st.button("ASK YOUR QUERY FOR NEW DATA"):
                    with st.spinner("Analyzing..."):
                        try:
                            
                            output = chain.invoke(input={"input_documents": docs, "question": query})
                            st.write(output["output_text"])
                        except Exception as e:
                            st.write(f"Failed to process the query: {e}")
                            
    #Additional Visualization mode
    elif operation_mode == "Visualize Your Data":
        if st.session_state.connection:
            table = st.selectbox("Choose a table", st.session_state.db.get_usable_table_names())
        
            if table:
                
                query = f"SELECT * FROM {table}"
                try:
                    df = pd.read_sql(query, st.session_state.connection)
                except ProgrammingError as e:
                    st.error(f"Table '{table}' does not exist in the database.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                
                st.write(f"Selected table: {table}")
                st.write(df.head(3))
                #Show the Generated image or code to streamlit using Pillow
                def base64_to_image(base64_string):
                    byte_data = base64.b64decode(base64_string)
                    return Image.open(BytesIO(byte_data))
                #Remove Random generation of Visualizations
                def set_random_seed(seed):
                    np.random.seed(seed)
                    random.seed(seed)

                set_random_seed(42)

                lida = Manager(text_gen=llm("openai"))
                textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-4o-mini", use_cache=True)

                summary = lida.summarize(data=df, summary_method="default", textgen_config=textgen_config)
                library = "seaborn"
                
                if "show_goals" not in st.session_state:
                    st.session_state.show_goals = False

                if "show_persona_input" not in st.session_state:
                    st.session_state.show_persona_input = False
                #Summarize Feature
                if st.sidebar.button("Summarize"):
                    st.write(summary)
                #Give Goals feature
                if st.sidebar.button("Give Goals"):
                    st.session_state.show_goals = True
                    st.session_state.show_persona_input = False
                
                if st.session_state.show_goals:
                    goals = lida.goals(summary=summary, n=3, textgen_config=textgen_config)
                    for goal in goals:
                        st.write(goal)
                    #Give Persona Based Goals feature
                    if st.sidebar.button("Give Goals based on persona"):
                        st.session_state.show_persona_input = True

                    if st.session_state.show_persona_input:
                        persona = st.text_input("Enter your persona")
                        if st.button("Generate Goals based on persona"):
                            if persona is not None:
                                personal_goals = lida.goals(summary=summary, n=2, persona=persona, textgen_config=textgen_config)
                                for pg in personal_goals:
                                    st.write(pg)
                            else:
                                st.error("Please enter a persona to generate goals based on it.")
                #Main Visualization block
                user_query = st.text_area("Query your Data to Generate Graph", height=200)
                if st.button("Generate Graph"):
                    if len(user_query) > 0:
                        set_random_seed(42)
                        st.info("Your Query: " + user_query)
                        charts = lida.visualize(summary=summary, goal=user_query, textgen_config=textgen_config,library=library)
                        st.write(f"Number of charts generated: {len(charts)}")
                        if charts and len(charts) > 0:
                            try:
                                # st.write(charts[0].code)
                                image_base64 = charts[0].raster
                                img = base64_to_image(image_base64)
                                st.image(img)
                            except IndexError:
                                st.error("No charts available to display.")
                        else:
                            st.error("No charts generated.")
footer_container.float("bottom: 0rem;")
                            

        
