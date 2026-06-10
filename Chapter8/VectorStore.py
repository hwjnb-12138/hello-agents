import math
from typing import List

class VectorItem():
    def __init__(self, document: str, embedding: List[float]):
        self.embedding = embedding
        self.document = document

class VectorStore():
    def __init__(self):
        self.vectorstore: List[VectorItem] = []

    def add(self, item: VectorItem):
        self.vectorstore.append(item)
    
    def search(self, query: List[float], top_k: int = 3):
        result = sorted(self.vectorstore, key=lambda x: self.cosine_similarity(query, x.embedding), reverse=True)
        return [x.document for x in result[:top_k]]
    
    def cosine_similarity(self, query: List[float], document: List[float]):
        dot_product = sum([q * d for q, d in zip(query, document)])
        query_magnitude = math.sqrt(sum([q ** 2 for q in query]))
        document_magnitude = math.sqrt(sum([d ** 2 for d in document]))
        if query_magnitude == 0 or document_magnitude == 0:
            return 0
        return dot_product / (query_magnitude * document_magnitude)
