from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from data_processor import preprocess
from typing import List

class TFIDFRetriever(BaseRetriever):
    """Custom retriever that integrates with your existing TF-IDF setup"""
    
    def __init__(self, vectorizer, tfidf_matrix, documents, k=10):
        super().__init__()
        # Store as private attributes to avoid field validation issues
        self._vectorizer = vectorizer
        self._tfidf_matrix = tfidf_matrix
        self._documents = documents
        self._k = k
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant to a query."""
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:self._k]
        return [self._documents[i] for i in top_indices if scores[i] > 0]

class CAGHybridRetriever:
    def __init__(self, processed_data):
        """Initialize with your existing processed data structure"""
        self.chunked_documents = processed_data['chunked_documents']
        self.vectorizer = processed_data['vectorizer']
        self.tfidf_matrix = processed_data['tfidf_matrix_chunks']
        
        # Initialize retrievers as class attributes
        self.bm25_retriever = None
        self.tfidf_retriever = None
        self.ensemble_retriever = None
        
        # Convert to LangChain documents
        self.langchain_docs = [
            Document(
                page_content=doc['text'],
                metadata={
                    'chunk_id': doc['chunk_id'],
                    'source_doc_id': doc['source_doc_id']
                }
            ) for doc in self.chunked_documents
        ]
        
        # Initialize retrievers
        self._setup_retrievers()
    
    def _setup_retrievers(self):
        """Setup BM25 and TF-IDF retrievers"""
        # BM25 retriever using LangChain
        self.bm25_retriever = BM25Retriever.from_documents(
            self.langchain_docs,
            k=10,
            preprocess_func=preprocess
        )
        
        # Custom TF-IDF retriever that uses your existing vectorizer
        self.tfidf_retriever = TFIDFRetriever(
            self.vectorizer, 
            self.tfidf_matrix, 
            self.langchain_docs
        )
        
        # Ensemble retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.tfidf_retriever],
            weights=[0.6, 0.4]
        )
    
    def retrieve(self, query, top_k=5):
        """Main retrieval method"""
        results = self.ensemble_retriever.invoke(query)
        return results[:top_k]
