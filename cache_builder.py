import pickle
import os
import time
from datetime import datetime, timedelta
from cachetools import TTLCache, cached
from tqdm import tqdm
from config import CACHE_FILE, LLM_MODEL_NAME, CHUNK_SIZE, GEMINI_API_KEY
from data_processor import initialize_and_preprocess

def load_cache():
    """Load cache data from disk"""
    if not os.path.exists(CACHE_FILE):
        return None
    
    try:
        with open(CACHE_FILE, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Handle both old and new cache formats
        if isinstance(cache_data, dict) and 'entries' in cache_data:
            # New format with metadata
            return cache_data['entries']
        else:
            # Old format - direct list
            return cache_data
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None

def build_cache():
    """Build cache using the advanced cache manager"""
    cache_manager = AdvancedCacheManager()
    cache_manager.build_cache_with_metadata()

class AdvancedCacheManager:
    def __init__(self, max_size=1000, ttl_hours=24):
        self.memory_cache = TTLCache(maxsize=max_size, ttl=ttl_hours * 3600)
        self.disk_cache_file = CACHE_FILE
        
    def load_cache(self):
        """Load cache from disk"""
        return load_cache()
        
    def build_cache_with_metadata(self):
        """Enhanced cache building with metadata and versioning"""
        print("Building enhanced CAG cache...")
        processed_data = initialize_and_preprocess()
        chunked_documents = processed_data['chunked_documents']

        cache_data = []
        for doc_chunk in tqdm(chunked_documents, desc="Preparing Enhanced Cache Entries"):
            chunk_id = doc_chunk['chunk_id']
            source_id = doc_chunk['source_doc_id']
            text = doc_chunk['text']
            
            cache_entry = {
                'chunk_id': chunk_id,
                'source_doc_id': source_id,
                'text_snippet': text[:500] + "..." if len(text) > 500 else text,
                'full_text': text,  # Store full text
                'created_at': datetime.now().isoformat(),
                'access_count': 0,
                'last_accessed': None,
                'text_hash': hash(text),  # For change detection
                'quality_score': self._calculate_quality_score(text),
                'semantic_keywords': self._extract_keywords(text),
                'chunk_size': len(text),
            }
            cache_data.append(cache_entry)

        # Add cache metadata
        cache_metadata = {
            'version': '2.0',
            'created_at': datetime.now().isoformat(),
            'total_chunks': len(cache_data),
            'source_documents': len(set(entry['source_doc_id'] for entry in cache_data))
        }
        
        final_cache = {
            'metadata': cache_metadata,
            'entries': cache_data
        }

        print(f"Enhanced cache building complete. Saving to {self.disk_cache_file}...")
        with open(self.disk_cache_file, 'wb') as f:
            pickle.dump(final_cache, f)
        print("Enhanced cache saved.")
        
    def _calculate_quality_score(self, text):
        """Calculate text quality score for prioritization"""
        # Simple heuristic - can be enhanced
        score = 0
        score += len(text.split()) / 100  # Word count factor
        score += text.count('.') / 10     # Sentence count factor
        return min(score, 1.0)
        
    def _extract_keywords(self, text):
        """Extract key terms for better indexing"""
        # Simple keyword extraction - enhance with NLP libraries
        words = text.lower().split()
        return [word for word in words if len(word) > 5][:10]
