from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.embeddings import SentenceTransformerEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import os



load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

def retrieve_from_pinecone(user_query="What is the syllabus about"):
    index_name = "test-index"
    index = pc.Index(index_name)
    
    #print("Index stats:", index.describe_index_stats())
    
    pinecone = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    context = pinecone.similarity_search(user_query)[:5]
    
    return context