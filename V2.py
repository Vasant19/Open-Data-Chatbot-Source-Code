import os
import pandas as pd
import streamlit as st
import pandasai
from pandasai import Agent
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set the backend to 'Agg' for non-interactive plotting
from  dotenv import load_dotenv
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
    left_co, cent_co, last_co = st.columns(3)
    with cent_co:
        st.image("cf.jpg")  
    
    st.title("Visualization and Data Analysis")
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
                try:
                    response = agent.chat(query)
                    
                    if isinstance(response, dict) and response.get("type") == "plot":
                        fig = response.get("value")
                        st.write("Generated Plot:")
                        st.pyplot(fig)
                    else:
                        formatted_response = format_response(response)
                        st.write("Response:")
                        st.write(formatted_response)
                except pd.errors.EmptyDataError:
                    st.error("No data found to analyze.")
                except pandasai.exceptions.NoResultFoundError:
                    st.error("No result returned from the analysis.")
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")

        # Example visualization generation based on user query
        if query.lower().startswith("plot"):
            try:
                query_parts = query.lower().split()
                plot_type = query_parts[1]
                column = query_parts[-1]

                fig, ax = plt.subplots()
                if plot_type == "lineplot":
                    data_information.plot(kind='line', y=column, ax=ax)
                elif plot_type == "barplot":
                    data_information.plot(kind='bar', y=column, ax=ax)
                elif plot_type == "scatterplot":
                    x_column = query_parts[-3]  # Assuming the query format is specific
                    data_information.plot(kind='scatter', x=x_column, y=column, ax=ax)
                else:
                    st.write("Plot type not recognized.")
                
                ax.set_title('Generated Plot')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"An error occurred while generating the plot: {str(e)}")

    else:
        st.write("Please upload a CSV file.")

if __name__ == "__main__":
    main()
