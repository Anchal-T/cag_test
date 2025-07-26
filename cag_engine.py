import os
from cache_builder import load_cache, AdvancedCacheManager
from retriever import CAGHybridRetriever
from llm_interface import get_llm_response_with_cache
from query_processor import QueryProcessor
from cache_analytics import CacheAnalytics
from adaptive_retriever import AdaptiveRetriever
from data_processor import process_new_document
import asyncio

class CAGEngine:
    def __init__(self, document_url=None):
        print("Initializing CAG Engine...")
        
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
            # Initialize data processing and retrieval
            self.processed_data = None  # Will be set per query
        
        # Initialize hybrid retriever
        self.retriever = None
        
        # Initialize adaptive retriever
        self.adaptive_retriever = None
        
        # Initialize query processor
        self.query_processor = QueryProcessor()
        
        # Initialize cache analytics
        self.cache_analytics = CacheAnalytics(self.cache_manager)
        
        print("CAG Engine initialized successfully!")
    
    def _setup_retriever_for_document(self, document_url):
        """Set up retriever for a specific document"""
        if not self.processed_data or self.processed_data['full_documents'][0]['id'] != document_url:
            self.processed_data = process_new_document(document_url)
        
        if not self.retriever:
            self.retriever = CAGHybridRetriever(self.processed_data)
            self.adaptive_retriever = AdaptiveRetriever(self.retriever)
        elif self.retriever.chunked_documents[0]['source_doc_id'] != document_url:
            # Reinitialize retriever for new document
            self.retriever = CAGHybridRetriever(self.processed_data)
            self.adaptive_retriever = AdaptiveRetriever(self.retriever)
    
    def generate_answer(self, query, document_url=None):
        """Main method to generate answers using integrated components"""
        import time
        start_time = time.time()
        
        try:
            # Set up retriever for the document
            if document_url:
                self._setup_retriever_for_document(document_url)
            elif not self.retriever:
                raise ValueError("No document URL provided and no retriever initialized")
            
            # Enhance query
            enhanced_terms = self.query_processor.enhance_query(query)
            intent = self.query_processor.detect_query_intent(query)
            
            # Use adaptive retrieval
            relevant_docs = self.adaptive_retriever.retrieve_with_adaptation(
                query, top_k=5
            )
            
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
            
            # Log analytics
            response_time = time.time() - start_time
            hit_type = 'hit' if relevant_entries else 'miss'
            
            for entry in relevant_entries:
                if entry.get('chunk_id'):
                    self.cache_analytics.log_cache_access(
                        entry['chunk_id'], query, response_time, hit_type
                    )
            
            return response
            
        except Exception as e:
            return f"Error generating response: {e}"
    
    def get_cache_report(self):
        """Get cache performance analytics"""
        return self.cache_analytics.generate_cache_report()
    
    def learn_from_feedback(self, query, feedback_score):
        """Learn from user feedback"""
        if self.retriever:
            relevant_docs = self.retriever.retrieve(query, top_k=5)
            self.adaptive_retriever.learn_from_feedback(query, relevant_docs, feedback_score)