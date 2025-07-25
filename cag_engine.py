# File: cag_engine.py
"""
Core logic for the Cache-Augmented Generation (CAG) system.
Manages loading cache, finding relevant cache entries, and interacting with the LLM.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from data_processor import initialize_and_preprocess
from cache_builder import load_cache
from config import CACHE_FILE

class CAGEngine:
    def __init__(self):
        print("Initializing CAG Engine...")
        self.processed_data = initialize_and_preprocess()
        self.cache_data = load_cache()

        if self.cache_data is None:
            raise FileNotFoundError(f"Cache file '{CACHE_FILE}' not found. Please run cache_builder.py first.")

        # Create a mapping from chunk_id to cache entry for quick lookup
        self.chunk_id_to_cache = {entry['chunk_id']: entry for entry in self.cache_data}

        # For retrieval, we'll use TF-IDF on the chunked documents (already processed)
        self.vectorizer_chunks = self.processed_data['vectorizer']
        self.tfidf_matrix_chunks = self.processed_data['tfidf_matrix_chunks']
        self.chunked_documents = self.processed_data['chunked_documents']

        print("CAG Engine initialized.")

    def retrieve_relevant_cache(self, query, top_k=1):
        """
        Retrieves the most relevant pre-computed cache entries based on the query.
        Uses TF-IDF on the original chunked text for retrieval.
        """
        print(f"Retrieving relevant knowledge cache for query: '{query}'...")
        if not query.strip():
            return []

        try:
            # Vectorize the query using the chunk vectorizer
            query_vec = self.vectorizer_chunks.transform([query])

            # Calculate cosine similarity between query and all chunk TF-IDF vectors
            scores = cosine_similarity(query_vec, self.tfidf_matrix_chunks).flatten()

            # Get top_k indices
            top_indices = np.argsort(scores)[::-1][:top_k]

            # Retrieve the corresponding cache entries
            relevant_entries = []
            for idx in top_indices:
                if scores[idx] > 0: # Only consider non-zero scores
                    chunk_info = self.chunked_documents[idx]
                    chunk_id = chunk_info['chunk_id']
                    cache_entry = self.chunk_id_to_cache.get(chunk_id)
                    if cache_entry:
                        relevant_entries.append({
                            'cache_entry': cache_entry,
                            'score': scores[idx],
                            'chunk_info': chunk_info # Include for context snippet if needed
                        })
                    else:
                        print(f"Warning: Cache entry for chunk_id {chunk_id} not found.")
            print(f"Found {len(relevant_entries)} relevant cache entry/entries.")
            return relevant_entries
        except Exception as e:
            print(f"Error during cache retrieval: {e}")
            return []

    def generate_answer(self, query):
        """Main function to process a query and generate an answer using CAG."""
        # 1. Retrieve relevant cache entries
        relevant_cache_results = self.retrieve_relevant_cache(query, top_k=2) # Get top 2

        if not relevant_cache_results:
             return "Sorry, I couldn't find relevant pre-loaded knowledge to answer your question."

        # Extract cache entries and optional snippets
        relevant_cache_entries = [res['cache_entry'] for res in relevant_cache_results]
        context_snippets = [res['chunk_info']['text'] for res in relevant_cache_results] # Use chunk text as snippet

        print("Generating answer using LLM with pre-loaded knowledge context...")
        # 2. Use LLM interface with cache (simulated by including text in prompt)
        from llm_interface import get_llm_response_with_cache # Import here
        answer = get_llm_response_with_cache(query, relevant_cache_entries, context_snippets)
        return answer
