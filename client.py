from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer

chatgpt_path = 'api_keys/chatgpt.txt'
gemini_path = 'api_keys/gemini.txt'
huggingface_path = 'api_keys/huggingface.txt'

def read_api_key(filepath):
    with open(filepath, 'r') as file:
        return file.read().strip()
    
def create_client(model='gpt'):
    if model == 'gpt':
        api_key = read_api_key(chatgpt_path)
        client = OpenAI(api_key=api_key)
    elif model == 'bert':
        client = SentenceTransformer('bert-base-nli-mean-tokens')
    elif model == 'gemini':
        pass
    elif model == 'huggingface':
        pass
    return client

def get_embedding_ada(text, chatgpt_client):
    response = chatgpt_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response.data[0].embedding).reshape(1, -1)

def get_embedding_small(text, chatgpt_client):
    response = chatgpt_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding).reshape(1, -1)

def get_embedding_large(text, chatgpt_client):
    response = chatgpt_client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return np.array(response.data[0].embedding).reshape(1, -1)

def get_embedding_bert(text, bert_model):
    return bert_model.encode([text])