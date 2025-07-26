import json
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

class CacheAnalytics:
    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
        self.access_log = []
        
    def log_cache_access(self, chunk_id, query, response_time, hit_type='hit'):
        """Log cache access for analytics"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'chunk_id': chunk_id,
            'query_hash': hash(query),
            'response_time': response_time,
            'hit_type': hit_type
        }
        self.access_log.append(log_entry)
        
    def generate_cache_report(self):
        """Generate comprehensive cache performance report"""
        if not self.access_log:
            return "No access data available"
            
        total_requests = len(self.access_log)
        hits = len([log for log in self.access_log if log['hit_type'] == 'hit'])
        hit_rate = hits / total_requests if total_requests > 0 else 0
        
        avg_response_time = np.mean([log['response_time'] for log in self.access_log])
        
        chunk_access_count = defaultdict(int)
        for log in self.access_log:
            chunk_access_count[log['chunk_id']] += 1
            
        most_accessed = sorted(chunk_access_count.items(), 
                             key=lambda x: x[1], reverse=True)[:5]
        
        report = {
            'cache_performance': {
                'hit_rate': f"{hit_rate:.2%}",
                'total_requests': total_requests,
                'average_response_time': f"{avg_response_time:.3f}s"
            },
            'most_accessed_chunks': most_accessed,
            'generated_at': datetime.now().isoformat()
        }
        
        return json.dumps(report, indent=2)
        
    def optimize_cache_size(self):
        """Suggest optimal cache size based on usage patterns"""
        # Analyze access patterns and suggest cache size
        if len(self.access_log) < 100:
            return "Insufficient data for optimization"
            
        # Implementation for cache size optimization
        unique_chunks = len(set(log['chunk_id'] for log in self.access_log))
        suggested_size = min(unique_chunks * 1.5, 2000)  # 50% buffer
        
        return f"Suggested cache size: {suggested_size} entries"