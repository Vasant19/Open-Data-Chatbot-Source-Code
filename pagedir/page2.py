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
from Callfuncs import *
from audio_recorder_streamlit import audio_recorder
from streamlit_float import float_init
from dotenv import load_dotenv

if st.session_state.role not in ["super-admin"]:
    st.warning("You do not have permission to view this page.")
    st.stop()

agent = st.session_state.get('agent', None)
if agent is None:
    st.error("Agent not initialized. Please return to the main page or click the connect button.")
    
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

        # Set seed
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
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_text(raw_text)

        document_search = FAISS.from_texts(texts, embeddings)
        query = st.text_input("Enter your Query for the pdf")
        docs = document_search.similarity_search(query)

        # Use ChatOpenAI from LangChain, NOT openai.OpenAI
        llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
        chain = load_qa_chain(llm=llm, chain_type="map_reduce")

        if st.button("ASK YOUR QUERY FOR NEW DATA"):
            with st.spinner("Analyzing..."):
                try:
                    output = chain.invoke({"input_documents": docs, "question": query})
                    st.write(output["output_text"])
                except Exception as e:
                    st.write(f"Failed to process the query: {e}")
