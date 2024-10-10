import os
import pandas as pd
import streamlit as st
from pandasai import Agent
from dotenv import load_dotenv
load_dotenv()

def load_data(file):
    df = pd.read_csv(file)
    return df

def format_response(response):
    if isinstance(response, int):
        return f"The answer is {response}."
    elif isinstance(response, pd.Timestamp):
        return f"The answer is {response.strftime('%Y-%m-%d %H:%M:%S')}."
    else:
        return str(response)
    
def main():
    st.title("Sample data analysis")

    uploaded_file = st.file_uploader("Upload your Data here", type=["csv"])

    if uploaded_file is not None:
        with st.spinner("Loading data..."):
            data_information = load_data(uploaded_file)

        agent = Agent(data_information)

        st.subheader("Uploaded data")
        st.write(data_information)

        query = st.text_input("Enter your question about the data: ")

    if st.button("Ask") and agent is not None:
            with st.spinner("Please Wait Analyzing..."):
                response = agent.chat(query)
            
            formatted_response = format_response(response)
            st.write("Response:")
    else:
        st.write("Please upload a CSV file.")

if __name__ == "__main__":
    main()
