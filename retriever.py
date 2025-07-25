# File: retriever.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from data_processor import preprocess

def search_tfidf(query, vectorizer, tfidf_matrix):
    """Performs TF-IDF search using a pre-fitted vectorizer and matrix."""
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    return scores

def search_bm25(query, bm25_model):
    """Performs BM25 search using a pre-built model."""
    tokenized_query = preprocess(query)
    scores = bm25_model.get_scores(tokenized_query)
    return scores

def hybrid_search(query, documents, bm25_model, vectorizer, tfidf_matrix, top_k=2, bm25_weight=0.6, tfidf_weight=0.4):
    """
    Performs a hybrid search by combining normalized BM25 and TF-IDF scores.
    """
    tfidf_scores = search_tfidf(query, vectorizer, tfidf_matrix)
    bm25_scores = search_bm25(query, bm25_model)

    if np.std(tfidf_scores) > 0:
        norm_tfidf = (tfidf_scores - np.min(tfidf_scores)) / (np.max(tfidf_scores) - np.min(tfidf_scores))
    else:
        norm_tfidf = np.zeros_like(tfidf_scores)
        
    if np.std(bm25_scores) > 0:
        norm_bm25 = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
    else:
        norm_bm25 = np.zeros_like(bm25_scores)
    
    combined_scores = (bm25_weight * norm_bm25) + (tfidf_weight * norm_tfidf)
    
    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    
    results = [{'doc': documents[i], 'score': combined_scores[i]} for i in top_indices]
    return results
