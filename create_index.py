import os
from pinecone import Pinecone,ServerlessSpec
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
# Initialize Pinecone with API key
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
# Define index name
index_name = "test-index"  # change if desired

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )


index = pc.Index(index_name)

# Print index statistics
print(index.describe_index_stats())