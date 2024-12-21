
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader  

import os
from pinecone import Pinecone
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import tempfile
from pathlib import Path
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings




load_dotenv()
#OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

embeddings = FastEmbedEmbeddings()

# Initialize Pinecone with API key
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Define Pinecone index
index_name = "testfinal"
index = pc.Index(index_name)
# Print index statistics
index.describe_index_stats()

vector_store = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

def process_and_upload_pdf(uploaded_file):
    # Create a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
        file_stem=temp_file.name


    # Load the PDF from the temporary file path
    loader = PyPDFLoader(temp_file_path)
    data = loader.load()
    
    # Split the document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    fragments = text_splitter.split_documents(data)
    
    # Extract texts and create metadata including the filename
    texts = [fragment.page_content for fragment in fragments]
    metadatas = [{"source": file_stem, "chunk_id": i} for i in range(len(fragments))]
    
    # Convert fragments into embeddings and store them in Pinecone
    vector_store.add_texts(
        texts=texts,
        metadatas=metadatas
    )
    
    # Optionally, remove the temporary file after processing
    os.remove(temp_file_path)

