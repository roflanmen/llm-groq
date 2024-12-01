from typing import List, Dict
from rank_bm25 import BM25Okapi
# from sentence_transformers import SentenceTransformer
import numpy as np

class Retriever:
    def __init__(self):
        self.chunks = []
        self.bm25 = None
        # self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        
    def add_documents(self, chunks: List[str]):
        self.chunks = chunks
        
        tokenized_chunks = [chunk.split() for chunk in chunks]
        print("loading bm25")
        self.bm25 = BM25Okapi(tokenized_chunks)
        print("bm25 loaded")
        # print("loading semantic model")
        # self.embeddings = self.semantic_model.encode(chunks)
        # print("semantic model loaded")
        
    def retrieve(self, query: str, 
                use_bm25: bool = True, 
                use_semantic: bool = True,
                top_k: int = 3) -> List[str]:
        results = []
        
        if use_bm25:
            tokenized_query = query.split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            bm25_top_k = np.argsort(bm25_scores)[-top_k:]
            results.extend([self.chunks[i] for i in bm25_top_k])
            
        # if use_semantic:
        #     query_embedding = self.semantic_model.encode(query)
        #     similarities = np.dot(self.embeddings, query_embedding)
        #     semantic_top_k = np.argsort(similarities)[-top_k:]
        #     results.extend([self.chunks[i] for i in semantic_top_k])
            
        return list(set(results))
