from pymilvus import MilvusClient
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import os
load_dotenv()

def get_client():
    client = MilvusClient(uri=os.getenv("MILVUS_URI"), token=os.getenv("MILVUS_TOKEN"))
    return client

def get_embedding():
    embeddings = HuggingFaceEmbeddings(
        model_name="./models/bge-base-zh",
        model_kwargs={
            #"device": "cuda" if torch.cuda.is_available() else "cpu",
            "device": "cpu",
            "trust_remote_code": True, 
        },
        encode_kwargs={"normalize_embeddings": True}, # 配置 IP 使用
    ) 
    return embeddings