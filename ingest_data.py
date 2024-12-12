
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
import os
import io
import tempfile


load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize Pinecone with API key
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Define Pinecone index
index_name = "test-index"
index = pc.Index(index_name)
# Print index statistics
index.describe_index_stats()

def process_and_upload_pdf(uploaded_file):
    # Create a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Load the PDF from the temporary file path
    loader = PyPDFLoader(temp_file_path)
    data = loader.load()
    
    # Split the document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    fragments = text_splitter.split_documents(data)
    
    # Print a sample of the combined fragments for debugging
    #print("Total Fragments:", len(fragments))
    sample_fragments = [str(fragment) for fragment in fragments[:3]]
    #print("Fragments sample:", [s.encode('ascii', errors='replace').decode() for s in sample_fragments])

    # Convert fragments into embeddings and store them in Pinecone
    pinecone_store = PineconeVectorStore.from_documents(
        fragments, embeddings, index_name=index_name
    )

    # Print index statistics after adding data
    #print("Index stats after upserting:", index.describe_index_stats())

    # Optionally, remove the temporary file after processing
    os.remove(temp_file_path)