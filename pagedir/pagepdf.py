from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from typing_extensions import Concatenate
import streamlit as st
import os
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(api_key=openai_api_key)

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    file_name = uploaded_file.name
    base_name = os.path.splitext(file_name)[0]
    
    pdf_reader = PdfReader(uploaded_file)
    num_pages = len(pdf_reader.pages)
    st.write(f"The filename is: {base_name}")
    st.write(f"The PDF has {num_pages} pages.")
    query = st.text_input("Enter your Query for the pdf")
    
    if st.button("Ask"):
        raw_text = ''
        for i, page in enumerate(pdf_reader.pages):
            content = page.extract_text()
            if content:
                raw_text += content
        
        text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 800,
        chunk_overlap  = 200,
        length_function = len,)
        texts = text_splitter.split_text(raw_text)
        
        document_search = FAISS.from_texts(texts, embeddings)
        
        chain = load_qa_chain(OpenAI(), chain_type="stuff")

        docs = document_search.similarity_search(query)
        output = chain.run(input_documents=docs, question=query)
        st.write(output)


    # Displaying first 500 characters from the first page
    # if num_pages > 0:
    #     first_page = pdf_reader.pages[0]
    #     text = first_page.extract_text()
    #     st.write("Sample text from the first page:")
    #     st.write(text[:500])
        
        
    