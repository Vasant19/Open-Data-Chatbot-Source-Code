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
    
def query_existing_data():
    st.header("Query Existing Data")

    # Retrieve the agent from session state
    agent = st.session_state.get('agent', None)
    if agent is None:
        st.error("Agent not initialized. Please return to the main page or click the connect button.")
        return

    with st.container():
        recorded_audio = audio_recorder(
            text="Voice Analysis: Click to Record",
            recording_color="#FF7F7F",
            neutral_color="#7B0323"
        )

    # User Input
    user_input = st.text_input("Enter your SQL query")
    query_button = st.button("ASK YOUR QUERY")

    if user_input and query_button:
        with st.spinner("Analyzing..."):
            try:
                sqlagentresponse = agent.invoke({"input": user_input})
                st.write(sqlagentresponse["output"])
            except Exception as e:
                st.write(f"Failed to process the query: {e}")

    elif recorded_audio:
        # Process recorded audio
        try:
            try:
                audio_file = "audio.mp3"
                with open(audio_file,"wb") as f:
                    f.write(recorded_audio)
                transcribed_text = speech_to_text(audio_file)
                agentTTS = agent.invoke({"input": transcribed_text})
                str_final_output = agentTTS["output"]
                response_audio_file = "audio_response.mp3"
                text_to_audio(str_final_output,response_audio_file)
                autoplay_audio((response_audio_file))
            except Exception as e:
                st.error("Could Not Autoplay the audio ")
            finally:
                st.write(agentTTS["output"])
                # st.write("Your Query: ", transcribed_text)
        except Exception as e:
            st.error(f"No Voice Detected, Retry please❗ ")

query_existing_data()