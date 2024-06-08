import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
import tempfile
import warnings
import os
from PyPDF2 import PdfReader
from urllib3.exceptions import NewConnectionError, MaxRetryError

def load_pdfs(file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name
        
        pdf_reader = PdfReader(tmp_file_path)
        total_pages = len(pdf_reader.pages)
        papers = []
        
        if total_pages <= 300:
            loader = PyPDFLoader(tmp_file_path)
            papers = loader.load()
        else:
            chunk_size = 100
            for start_page in range(0, total_pages, chunk_size):
                end_page = min(start_page + chunk_size, total_pages)
                chunk_path = f"{tmp_file_path}_chunk_{start_page}_{end_page}.pdf"
                with open(chunk_path, "wb") as chunk_file:
                    pdf_writer = PyPDF2.PdfWriter()
                    for page_num in range(start_page, end_page):
                        pdf_writer.add_page(pdf_reader.pages[page_num])
                    pdf_writer.write(chunk_file)
                
                loader = PyPDFLoader(chunk_path)
                papers.extend(loader.load())
                
        return papers
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return []

def split_documents(papers):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=1000
        )
        paper_chunks = text_splitter.split_documents(papers)
        return paper_chunks
    except Exception as e:
        st.error(f"Error splitting documents: {e}")
        return []

def create_vector_store(paper_chunks, embedding_function):
    try:
        qdrant = Qdrant.from_documents(
            paper_chunks,
            embedding_function,
            location=":memory:",
            collection_name="document_embeddings",
        )
        return qdrant
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

model_name = "BAAI/bge-base-en"
encode_kwargs = {'normalize_embeddings': True}
embedding_function = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'mps'},
    encode_kwargs=encode_kwargs
)

st.title("PDF Question Answering App")

uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    papers = []
    for uploaded_file in uploaded_files:
        papers.extend(load_pdfs(uploaded_file))
    
    if papers:
        paper_chunks = split_documents(papers)
        if paper_chunks:
            vector_store = create_vector_store(paper_chunks, embedding_function)
            if vector_store:
                retriever = vector_store.as_retriever(search_kwargs={"k": 2})

                try:
                    llm = Ollama(model="llama3", temperature=0.6, num_predict=300)
                except (NewConnectionError, MaxRetryError) as e:
                    st.error(f"Error initializing LLM: {e}")
                    llm = None

                if llm:
                    prompt_template = """
                    Use the following pieces of information to answer the user's question.
                    If you don't know the answer, just say that you don't know, don't try to make up an answer.

                    Context: {context}

                    Question: {question}

                    Answer the question and provide additional helpful information,
                    based on the pieces of information, if applicable. Be succinct.
                    Responses should be properly formatted to be easily read.
                    """

                    prompt = PromptTemplate(
                        template=prompt_template, input_variables=["context", "question"]
                    )

                    qa = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=retriever,
                        chain_type_kwargs={"prompt": prompt, "verbose": True},
                        return_source_documents=True
                    )

                    query = st.text_input("Enter your query")

                    if query:
                        with st.spinner('Processing...'):
                            import time
                            start_time = time.time()
                            try:
                                response = qa.invoke(query)
                                end_time = time.time()
                                st.write("### Response:")
                                st.write(response)
                                st.write("### Source Documents:")
                                for doc in response['source_documents']:
                                    st.write(doc)
                                st.write(f"Retrieval time: {end_time - start_time} seconds")
                            except Exception as e:
                                st.error(f"Error processing query: {e}")

