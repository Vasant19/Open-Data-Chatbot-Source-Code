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
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)
float_init()
footer_container = st.container()
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
                user_query = st.text_input("Query your Data to Generate Graph")
                graphbutton = st.button("Generate Graph")

                with footer_container:
                    audio_recorded = audio_recorder(text="Voice Analysis: Click to Record",recording_color="#FF7F7F",neutral_color="#7B0323")
                    if audio_recorded:
                        try:
                            file_audio = "visualize.mp3"
                            with open(file_audio,"wb") as f:
                                f.write(audio_recorded)
                            tt = speech_to_text(file_audio)
                        except Exception as e:
                            st.error("Voice length too small ❗")
            try:
                if len(user_query) > 0 and graphbutton:
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
      
                elif tt and graphbutton:
                    set_random_seed(42)
                    st.info("Your Query: " + tt)
                    charts = lida.visualize(summary=summary, goal=tt, textgen_config=textgen_config,library=library)
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
            except Exception as e:
                st.error("No Transcript or Input Detected")  
footer_container.float("bottom: 0rem;")
