# from langchain_community.embeddings.ollama import OllamaEmbeddings
# from langchain_community.embeddings.bedrock import BedrockEmbeddings
#from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os
from dotenv import load_dotenv

import os

load_dotenv()



def get_embedding_function():
    #os.environ['OPENAI_API_KEY'] = 
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
#     embeddings = AzureOpenAIEmbeddings(
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     #azure_deployment="${}",
#     openai_api_version="2025-01-01-preview",  # or whatever version your deployment uses
# )
    return embeddings
