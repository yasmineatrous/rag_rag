from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.embeddings import SentenceTransformerEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings('ignore')
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings




load_dotenv()

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))


embeddings = FastEmbedEmbeddings()
def retrieve_from_pinecone(user_query="What is the syllabus about"):
    index_name = "testfinal"
    index = pc.Index(index_name)
    
    #print("Index stats:", index.describe_index_stats())
    
    pinecone = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    context = pinecone.similarity_search(user_query)[:5]
    
    return context
