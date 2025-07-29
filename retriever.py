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
    Custom retriever that uses a pre-built Annoy index for semantic search.
    The 'index' and 'k' fields are declared at the class level to comply with
    LangChain's Pydantic-based BaseRetriever.
    """
    index: Annoy
    k: int = 10

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get documents relevant to a query using the Annoy index for semantic search.
        This method is called by the LangChain framework.
        """
        # The similarity_search method of the Annoy vector store performs the semantic search.
        return self.index.similarity_search(query, k=self.k)

class CAGHybridRetriever:
    def __init__(self, processed_data):
        """
        Initialize the hybrid retriever with your existing processed data.
        This retriever combines a keyword-based search (BM25) and a semantic search (Annoy).
        """
        self.chunked_documents = processed_data['chunked_documents']
        
        # Initialize retrievers as class attributes
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.annoy_retriever: Optional[AnnoyRetriever] = None
        self.ensemble_retriever: Optional[EnsembleRetriever] = None
        
        # Convert to LangChain documents, which are required by the retrievers.
        self.langchain_docs = [
            Document(
                page_content=doc['text'],
                metadata={
                    'chunk_id': doc['chunk_id'],
                    'source_doc_id': doc['source_doc_id']
                }
            ) for doc in self.chunked_documents
        ]
        
        # Initialize the individual retrievers and the ensemble retriever.
        self._setup_retrievers(processed_data['annoy_index_file'])
    
    def _setup_retrievers(self, annoy_index_file):
        """
        Set up the BM25 (keyword) and Annoy (semantic) retrievers.
        """
        # 1. BM25 Retriever (Keyword-based search)
        # This retriever is good for finding documents with exact keyword matches.
        self.bm25_retriever = BM25Retriever.from_documents(
            self.langchain_docs,
            k=10,
            preprocess_func=preprocess
        )
        
        # 2. Annoy Retriever (Semantic Search)
        # This retriever finds documents that are semantically similar to the query,
        # even if they don't contain the exact keywords.
        
        # Load the embeddings model. This is the same model used to create the Annoy index.
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        # Load the local Annoy index from the file created by data_processor.py.
        # allow_dangerous_deserialization is needed to load the index from a pickle file.
        annoy_index = Annoy.load_local(annoy_index_file, embeddings, allow_dangerous_deserialization=True)
        
        # Instantiate our custom AnnoyRetriever, which is a LangChain-compatible retriever.
        self.annoy_retriever = AnnoyRetriever(index=annoy_index)
        
        # 3. Ensemble Retriever (Hybrid Search)
        # This retriever combines the results from both the BM25 and Annoy retrievers.
        # The 'weights' parameter controls the contribution of each retriever to the final score.
        # Here, we are giving equal weight to both keyword and semantic search.
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.annoy_retriever],
            weights=[0.5, 0.5] # You can tune these weights for better performance.
        )
    
    def retrieve(self, query, top_k=5):
        """
        The main retrieval method. It uses the ensemble retriever to get the best of both
        keyword and semantic search.
        """
        if self.ensemble_retriever is None:
            raise ValueError("Ensemble retriever has not been initialized.")
        
        # The 'invoke' method of the ensemble retriever runs the query against both retrievers
        # and combines the results based on the weights.
        results = self.ensemble_retriever.invoke(query)
        
        return results[:top_k]