import streamlit as st
from Callfuncs import *
from streamlit_lottie import st_lottie
import time
from menu import menu_with_redirect
menu_with_redirect()
# Verify the user's role
if st.session_state.role not in ["super-admin"]:
    st.warning("You do not have permission to view this page.")
    st.stop()

st.write(f"You are currently logged with the role of {st.session_state.role}.")
st.title("🙏WELCOME TO TICASUK🙏")
welcomemessage = "Hello!!....Welcome to 'TICASUK'.... I Humbly welcome you. THIS is the home page of the application. In order to start ,you need to Press the connect button and choose a mode first and THEN ask me your questions! For further help you can press the menu button on top right and click 'Get help' button. IF..you face any issues or would Like to give feedback, click on the Report a bug button. Thank you for using 'Ticasuk',and, Enjoy querying!!!!"
wma = "welcomemessage.mp3"
text_to_audio(welcomemessage,wma)
autoplay_audio((wma))
welcome = "https://lottie.host/c0ca3d28-4906-4f32-a311-bd967ab7ca08/vC7930dUQ8.json"
st.lottie(welcome)