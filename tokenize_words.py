import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import json
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Download NLTK data (only need to do this once) ---
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# --- File path for our "database" ---
PERSISTENCE_FILE = "rag_data.pkl"

def preprocess(text):
    """Cleans, tokenizes, removes stop words, and lemmatizes text."""
    stop_words = set(stopwords.words('english'))
    punct = set(string.punctuation)
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    filtered_tokens = [
        lemmatizer.lemmatize(word) for word in tokens 
        if word.isalpha() and word not in stop_words and word not in punct
    ]
    return filtered_tokens

def initialize_and_preprocess():
    """
    Loads data from persistence file if it exists. 
    Otherwise, processes documents and saves them.
    """
    if os.path.exists(PERSISTENCE_FILE):
        print("Loading pre-processed data from disk...")
        with open(PERSISTENCE_FILE, 'rb') as f:
            data = pickle.load(f)
        return data

    print("No pre-processed data found. Starting one-time processing...")
    
    # Load your sample papers from arxiv_papers.json
    with open('arxiv_papers.json', 'r', encoding='utf-8') as f:
        sample_papers = json.load(f)
    
    # 1. Preprocess for BM25
    print("Processing documents for BM25...")
    tokenized_corpus = [preprocess(doc['content']) for doc in sample_papers]

    # 2. Preprocess for TF-IDF
    print("Processing documents for TF-IDF...")
    raw_texts = [doc['content'] for doc in sample_papers]
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(raw_texts)
    
    # 3. Bundle everything and save to disk
    data_to_persist = {
        "sample_papers": sample_papers,
        "tokenized_corpus": tokenized_corpus,
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix
    }
    
    print(f"Processing complete. Saving to {PERSISTENCE_FILE}...")
    with open(PERSISTENCE_FILE, 'wb') as f:
        pickle.dump(data_to_persist, f)
        
    return data_to_persist

# Example usage:
if __name__ == "__main__":
    data = initialize_and_preprocess()
