from dotenv import load_dotenv
from openai import OpenAI
import os
import base64
import streamlit as st

load_dotenv("openai_api_key")
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

def speech_to_text(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(model="whisper-1",file=audio_file,response_format="text", language="en")  
        return transcript

def text_to_audio(text, audio_path):
    # formatted_text = format_text_for_tts(text)
    response = client.audio.speech.create(model="tts-1",voice="nova",input=text)#input=formatted_text
    response.stream_to_file(audio_path)
    
def autoplay_audio(audio_file):
    with open(audio_file,"rb") as audio_file:
        audio_reader = audio_file.read()
    base64_audio=base64.b64encode(audio_reader).decode("utf-8")
    audio_html = f'<audio src="data:audio/mp3;base64,{base64_audio}" controls autoplay>'
    st.markdown(audio_html,unsafe_allow_html=True)
