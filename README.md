import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pytesseract
from PIL import Image
import io
import pdf2image
import docx
import textract
import cv2
import numpy as np

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_text_from_file(file):
    text = ""
    if file.type == "application/pdf":
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file.type == "image/jpeg" or file.type == "image/png":
        img = Image.open(file)
        text += pytesseract.image_to_string(img)
    elif file.type == "application/msword" or file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text
    elif file.type == "text/plain":
        text = file.read().decode("utf-8")
    else:
        text = textract.process(file.read()).decode("utf-8")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=2000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "Ramu doesn't know shit!! ", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain.invoke(
        {"input_documents": docs, "question": user_question}
       , return_only_outputs=True)

    print(response)
    st.write("Answer: ", response["output_text"])

def main():
    st.set_page_config("Chat File")
    st.header("Welcome to unversity policy Q&A system!! ")

    user_question = st.text_input("what do you want to know ")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        file = st.file_uploader("Upload your file", type=["pdf", "jpg", "jpeg", "png", "docx", "txt", "odt"])
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                text = get_text_from_file(file)
                text_chunks = get_text_chunks(text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
