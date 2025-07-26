from retriever import CAGHybridRetriever
from data_processor import initialize_and_preprocess
from cache_builder import load_cache
from config import CACHE_FILE
from llm_interface import get_llm_response_with_cache

class CAGEngine:
    def __init__(self):
        print("Initializing CAG Engine...")
        self.processed_data = initialize_and_preprocess()
        self.cache_data = load_cache()
        
        if self.cache_data is None:
            raise FileNotFoundError(f"Cache file '{CACHE_FILE}' not found.")
        
        # Initialize hybrid retriever
        self.hybrid_retriever = CAGHybridRetriever(self.processed_data)
        
        # Keep cache mapping for backwards compatibility
        self.chunk_id_to_cache = {entry['chunk_id']: entry for entry in self.cache_data}
        
    
    def retrieve_relevant_cache(self, query, top_k=2):
        """Updated to use LangChain hybrid retriever"""
        print(f"Retrieving relevant knowledge using hybrid search...")
        
        try:
            # Use LangChain ensemble retriever
            retrieved_docs = self.hybrid_retriever.retrieve(query, top_k)
            
            relevant_entries = []
            for doc in retrieved_docs:
                chunk_id = doc.metadata['chunk_id']
                cache_entry = self.chunk_id_to_cache.get(chunk_id)
                
                if cache_entry:
                    relevant_entries.append({
                        'cache_entry': cache_entry,
                        'document': doc,
                        'text': doc.page_content
                    })
            
            print(f"Found {len(relevant_entries)} relevant cache entries.")
            return relevant_entries
            
        except Exception as e:
            print(f"Error during hybrid retrieval: {e}")
            # Fallback to original method
            return self._fallback_retrieval(query, top_k)

    def generate_answer(self, query):
        """Generate an answer using retrieved cache entries and LLM"""
        # Retrieve relevant cache entries
        relevant_entries = self.retrieve_relevant_cache(query)
        
        # Extract cache entries for the LLM interface
        cache_entries = [entry['cache_entry'] for entry in relevant_entries]
        
        # Generate response using LLM
        return get_llm_response_with_cache(query, cache_entries)

    def _fallback_retrieval(self, query, top_k):
        """Fallback retrieval method - implement based on your original logic"""
        # This method should implement your original retrieval logic
        # For now, returning empty list
        print("Using fallback retrieval method...")
        return []