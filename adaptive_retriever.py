from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class AdaptiveRetriever:
    def __init__(self, base_retriever):
        self.base_retriever = base_retriever
        self.query_patterns = defaultdict(list)
        self.success_metrics = defaultdict(float)
        self.adaptation_threshold = 0.7
        
        # Initialize simple vectorizer for query embeddings
        self.query_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self._is_fitted = False
        
    def retrieve_with_adaptation(self, query, top_k=5, user_feedback=None):
        """Adaptive retrieval that learns from user feedback"""
        try:
            # Get base retrieval results
            base_results = self.base_retriever.retrieve(query, top_k * 2)
            
            # Apply learned adaptations
            adapted_results = self._apply_adaptations(query, base_results)
            
            # Re-rank based on historical success
            reranked_results = self._rerank_by_success(query, adapted_results)
            
            # Store query pattern for learning
            self._store_query_pattern(query, reranked_results[:top_k])
            
            return reranked_results[:top_k]
        except Exception as e:
            print(f"Error in adaptive retrieval: {e}")
            # Fallback to base retriever
            return self.base_retriever.retrieve(query, top_k)
        
    def _apply_adaptations(self, query, results):
        """Apply learned patterns to improve retrieval"""
        try:
            for result in results:
                # Boost score based on historical success
                chunk_id = result.metadata.get('chunk_id', 'unknown')
                historical_score = self.success_metrics.get(chunk_id, 0.5)
                result.metadata['adapted_score'] = historical_score
        except Exception as e:
            print(f"Error applying adaptations: {e}")
            
        return results
    
    def _rerank_by_success(self, query, results):
        """Re-rank results based on historical success"""
        try:
            # Sort by adapted score if available, otherwise keep original order
            def get_score(result):
                return result.metadata.get('adapted_score', 0.5)
            
            return sorted(results, key=get_score, reverse=True)
        except Exception as e:
            print(f"Error re-ranking results: {e}")
            return results
    
    def _store_query_pattern(self, query, results):
        """Store query patterns for future learning"""
        try:
            pattern_key = self._get_query_pattern(query)
            chunk_ids = [r.metadata.get('chunk_id') for r in results if r.metadata.get('chunk_id')]
            self.query_patterns[pattern_key].extend(chunk_ids)
        except Exception as e:
            print(f"Error storing query pattern: {e}")
    
    def _get_query_pattern(self, query):
        """Extract pattern from query for categorization"""
        query_lower = query.lower()
        if any(word in query_lower for word in ['coverage', 'covered', 'include']):
            return 'coverage'
        elif any(word in query_lower for word in ['claim', 'file', 'submit']):
            return 'claim'
        elif any(word in query_lower for word in ['premium', 'cost', 'price']):
            return 'pricing'
        elif any(word in query_lower for word in ['exclusion', 'not covered', 'exclude']):
            return 'exclusion'
        else:
            return 'general'
    
    def _get_query_embedding(self, query):
        """Get query embedding for similarity calculations"""
        try:
            if not self._is_fitted:
                # Fit on a sample query if not fitted yet
                self.query_vectorizer.fit([query])
                self._is_fitted = True
            
            return self.query_vectorizer.transform([query]).toarray()[0]
        except Exception as e:
            print(f"Error getting query embedding: {e}")
            return np.zeros(100)  # Return zero vector as fallback
        
    def learn_from_feedback(self, query, retrieved_chunks, feedback_score):
        """Learn from user feedback to improve future retrievals"""
        try:
            for chunk in retrieved_chunks:
                chunk_id = chunk.metadata.get('chunk_id')
                if chunk_id:
                    current_score = self.success_metrics[chunk_id]
                    # Exponential moving average for score updates
                    self.success_metrics[chunk_id] = 0.7 * current_score + 0.3 * feedback_score
        except Exception as e:
            print(f"Error learning from feedback: {e}")