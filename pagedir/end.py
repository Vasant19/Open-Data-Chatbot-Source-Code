import streamlit as st

st.title("🙏A HEARTFELT THANK YOU TO ALL SEBASTIAN, CURTIS, JOSHUA AND SEAN FOR SUPPORTING US THROUGHOUT THESE MONTHS. WE PRESENT A FINAL VIDEO TO SHOWCASE 'WHAT IS CANADA' : Amitha, Manpreet, Daniel, Vasant 🙏")
video_file = open("Welcome home to Canada.mp4", "rb")
video_bytes = video_file.read()
st.video(video_bytes, autoplay=True)