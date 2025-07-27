from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List, Optional
from data_processor import preprocess
from langchain_community.vectorstores import Annoy
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_NAME

class AnnoyRetriever(BaseRetriever):
    """
    Custom retriever that uses a pre-built Annoy index.
    The 'index' and 'k' fields are declared at the class level to comply with
    LangChain's Pydantic-based BaseRetriever.
    """
    index: Annoy
    k: int = 10

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant to a query using the Annoy index."""
        return self.index.similarity_search(query, k=self.k)

class CAGHybridRetriever:
    def __init__(self, processed_data):
        """Initialize with your existing processed data structure"""
        self.chunked_documents = processed_data['chunked_documents']
        
        # Initialize retrievers as class attributes
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.annoy_retriever: Optional[AnnoyRetriever] = None
        self.ensemble_retriever: Optional[EnsembleRetriever] = None
        
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
        self._setup_retrievers(processed_data['annoy_index_file'])
    
    def _setup_retrievers(self, annoy_index_file):
        """Setup BM25 and Annoy retrievers"""
        # BM25 retriever using LangChain
        self.bm25_retriever = BM25Retriever.from_documents(
            self.langchain_docs,
            k=10,
            preprocess_func=preprocess
        )
        
        # Annoy retriever
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        # Load the local Annoy index
        annoy_index = Annoy.load_local(annoy_index_file, embeddings, allow_dangerous_deserialization=True)
        # Instantiate our custom retriever, passing the loaded index as a keyword argument
        self.annoy_retriever = AnnoyRetriever(index=annoy_index)
        
        # Ensemble retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.annoy_retriever],
            weights=[0.5, 0.5] # Adjusted weights for a more balanced retrieval
        )
    
    def retrieve(self, query, top_k=5):
        """Main retrieval method"""
        if self.ensemble_retriever is None:
            raise ValueError("Ensemble retriever has not been initialized.")
        results = self.ensemble_retriever.invoke(query)
        return results[:top_k]
