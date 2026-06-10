import os
import dotenv
import requests
from VectorStore import VectorItem, VectorStore

dotenv.load_dotenv()

class Embedding():
    def __init__(self, model: str):
        self.model = model
        self.vectorstore = VectorStore()

    def get_embedding(self, input: str):
        payload = {
            "model": self.model,
            "input": input,
        }
        headers = {
            "Authorization": f"Bearer {os.getenv('EMBEDDING_API_KEY')}",
            "Content-Type": "application/json",
        }
        response = requests.post(os.getenv('EMBEDDING_BASE_URL'), headers=headers, json=payload)
        data = response.json()
        return data["data"][0]["embedding"]
    
    def get_document_embedding(self, document: str):
        res = self.get_embedding(document)
        self.vectorstore.add(VectorItem(document, res))
        return res
    
    def search(self, query: str, top_k: int = 3):
        query_embedding = self.get_embedding(query)
        return self.vectorstore.search(query_embedding, top_k)
