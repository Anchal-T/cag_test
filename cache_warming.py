import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
import numpy as np
from datetime import datetime
from llm_interface import get_llm_response_with_cache

class CacheWarmer:
    def __init__(self, cache_manager, retriever=None):
        self.cache_manager = cache_manager
        self.retriever = retriever
        self.common_queries = [
            "What is the policy coverage?",
            "What are the exclusions?",
            "How to file a claim?",
            "What is the premium amount?",
            "What are the terms and conditions?"
        ]
        
    async def warm_cache_with_common_queries(self):
        """Pre-compute responses for common queries"""
        print("Warming cache with common queries...")
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            tasks = []
            for query in self.common_queries:
                task = executor.submit(self._precompute_query_response, query)
                tasks.append(task)
                
            for task in tasks:
                try:
                    result = task.result()
                    if result:
                        print(f"Pre-computed response for: {result['query'][:50]}...")
                except Exception as e:
                    print(f"Error warming cache: {e}")
                    
    def _precompute_query_response(self, query):
        """Precompute and cache query responses"""
        try:
            if self.retriever:
                # Use retriever to get relevant cache entries
                relevant_docs = self.retriever.retrieve(query, top_k=5)
                relevant_entries = [{'text_snippet': doc.page_content} for doc in relevant_docs]
            else:
                # Fallback to cache manager
                cache_entries = self.cache_manager.load_cache() if hasattr(self.cache_manager, 'load_cache') else []
                relevant_entries = cache_entries[:5] if cache_entries else []
            
            # Generate response using LLM interface
            response = get_llm_response_with_cache(query, relevant_entries)
            cache_key = f"precomputed_{hash(query)}"
            
            precomputed_entry = {
                'query': query,
                'response': response,
                'computed_at': datetime.now().isoformat(),
                'type': 'precomputed',
                'cache_key': cache_key
            }
            
            # Store in cache manager if it has a memory cache
            if hasattr(self.cache_manager, 'memory_cache'):
                self.cache_manager.memory_cache[cache_key] = precomputed_entry
            
            return precomputed_entry
        except Exception as e:
            print(f"Error precomputing query {query}: {e}")
            return None