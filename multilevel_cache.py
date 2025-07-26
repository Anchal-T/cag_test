from cachetools import LRUCache, TTLCache
import redis
import json
from config import CACHE_FILE  # Added missing import

class MultiLevelCache:
    def __init__(self):
        # L1: In-memory LRU cache for hot data
        self.l1_cache = LRUCache(maxsize=100)
        
        # L2: In-memory TTL cache for recent data
        self.l2_cache = TTLCache(maxsize=500, ttl=3600)  # 1 hour TTL
        
        # L3: Redis cache for distributed caching (optional)
        try:
            self.l3_cache = redis.Redis(host='localhost', port=6379, db=0)
        except:
            self.l3_cache = None
            
        # L4: Disk cache (existing pickle file)
        self.l4_cache_file = CACHE_FILE
        
    def get(self, key):
        """Multi-level cache get with promotion"""
        # Check L1 first
        if key in self.l1_cache:
            return self.l1_cache[key]
            
        # Check L2
        if key in self.l2_cache:
            value = self.l2_cache[key]
            # Promote to L1
            self.l1_cache[key] = value
            return value
            
        # Check L3 (Redis)
        if self.l3_cache:
            try:
                value = self.l3_cache.get(key)
                if value:
                    value = json.loads(value)
                    # Promote to L2 and L1
                    self.l2_cache[key] = value
                    self.l1_cache[key] = value
                    return value
            except:
                pass
                
        # Check L4 (Disk) - implement disk lookup
        return None
        
    def set(self, key, value):
        """Set value in appropriate cache levels"""
        # Always set in L1
        self.l1_cache[key] = value
        
        # Set in L2 for persistence
        self.l2_cache[key] = value
        
        # Optionally set in L3 (Redis)
        if self.l3_cache:
            try:
                self.l3_cache.setex(key, 3600, json.dumps(value))
            except:
                pass