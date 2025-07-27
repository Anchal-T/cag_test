import os
from cache_builder import load_cache, AdvancedCacheManager
from retriever import CAGHybridRetriever
from llm_interface import get_llm_response_with_cache
from query_processor import QueryProcessor
from data_processor import process_new_document
import asyncio
from typing import Optional

class CAGEngine:
    def __init__(self, document_url=None):
        
        # Initialize cache manager
        self.cache_manager = AdvancedCacheManager()
        
        # Process document dynamically if URL provided
        if document_url:
            print(f"Processing document: {document_url}")
            self.processed_data = process_new_document(document_url)
        else:
            # Load cache if no document URL provided
            self.cache_entries = load_cache()
            if not self.cache_entries:
                raise FileNotFoundError("Cache not found. Please build cache first.")
            self.processed_data = None  # Will be set per query
        
        # Initialize hybrid retriever with proper type annotation
        self.retriever: Optional[CAGHybridRetriever] = None        
        # Initialize query processor
        self.query_processor = QueryProcessor()        
        print("CAG Engine initialized successfully!")
    
    def _setup_retriever_for_document(self, document_url):
        """Set up retriever for a specific document"""
        if not self.processed_data or self.processed_data['full_documents'][0]['id'] != document_url:
            self.processed_data = process_new_document(document_url)
        
        if not self.retriever:
            self.retriever = CAGHybridRetriever(self.processed_data)
        elif self.retriever.chunked_documents[0]['source_doc_id'] != document_url:
            # Reinitialize retriever for new document
            self.retriever = CAGHybridRetriever(self.processed_data)
    
    def generate_answer(self, query, document_url=None):
        try:
            # Set up retriever for the document
            if document_url:
                self._setup_retriever_for_document(document_url)
            elif not self.retriever:
                raise ValueError("No document URL provided and no retriever initialized")
            
            # Retrieve relevant documents using the retriever
            if self.retriever is None:
                raise ValueError("Retriever is not initialized. Please provide a valid document_url or initialize the retriever.")
            
            relevant_docs = self.retriever.retrieve(query)
                    
            # Convert to cache entry format
            relevant_entries = []
            for doc in relevant_docs:
                entry = {
                    'text_snippet': doc.page_content,
                    'chunk_id': doc.metadata.get('chunk_id'),
                    'source_doc_id': doc.metadata.get('source_doc_id')
                }
                relevant_entries.append(entry)
            
            # Generate response using LLM
            response = get_llm_response_with_cache(query, relevant_entries)
    
            return response
            
        except Exception as e:
            return f"Error generating response: {e}"
    
    def generate_batch_answers(self, queries, document_url=None):
        responses = []
        
        try:
            # Set up retriever for the document (done once for all queries)
            if document_url:
                self._setup_retriever_for_document(document_url)
            elif not self.retriever:
                raise ValueError("No document URL provided and no retriever initialized")
            
            
            # Process each query
            for i, query in enumerate(queries):
                
                try:
                    # Retrieve relevant documents using the retriever
                    relevant_docs = self.retriever.retrieve(query)
                    
                    # Convert to cache entry format
                    relevant_entries = []
                    
                    for doc in relevant_docs:
                        entry = {
                            'text_snippet': doc.page_content,
                            'chunk_id': doc.metadata.get('chunk_id'),
                            'source_doc_id': doc.metadata.get('source_doc_id')
                        }
                        relevant_entries.append(entry)
                    
                    # Generate response using LLM
                    response = get_llm_response_with_cache(query, relevant_entries)
                    responses.append(response)
                                                        
                except Exception as e:
                    responses.append(f"Error processing query '{query}': {e}")
            
        except Exception as e:
            return [f"Error in batch processing: {e}"] * len(queries)

