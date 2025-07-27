import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import pickle
import os
import requests
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from config import PERSISTENCE_FILE, CHUNK_SIZE, CHUNK_OVERLAP, DOCUMENT_CACHE_FILE
from tqdm import tqdm
import re
from datetime import datetime, timedelta

# --- Download NLTK data (only need to do this once) ---
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Document cache with expiration (7 days)
DOCUMENT_CACHE_EXPIRY = timedelta(days=7)

def load_document_cache():
    """Load document cache from disk"""
    if os.path.exists(DOCUMENT_CACHE_FILE):
        try:
            with open(DOCUMENT_CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
                # Check if cache format is current
                if isinstance(cache, dict) and 'documents' in cache and isinstance(cache['documents'], dict):
                    return cache
                else:
                    # Old format or corrupted, create new cache
                    return {'documents': {}, 'last_updated': datetime.now()}
        except (pickle.PickleError, EOFError, Exception):
            # Corrupted cache, create new one
            return {'documents': {}, 'last_updated': datetime.now()}
    return {'documents': {}, 'last_updated': datetime.now()}

def save_document_cache(cache):
    """Save document cache to disk"""
    try:
        with open(DOCUMENT_CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)
    except Exception as e:
        print(f"Warning: Could not save document cache: {e}")

def is_cache_valid(timestamp):
    """Check if cache entry is still valid"""
    if isinstance(timestamp, datetime):
        return datetime.now() - timestamp < DOCUMENT_CACHE_EXPIRY
    return False

def get_cached_document(url):
    """Retrieve document from cache if available and valid"""
    cache = load_document_cache()
    # Ensure cache has the correct structure
    if not isinstance(cache, dict) or 'documents' not in cache or not isinstance(cache['documents'], dict):
        return None
        
    documents = cache['documents']
    if url in documents:
        doc_entry = documents[url]
        if isinstance(doc_entry, dict) and 'timestamp' in doc_entry and 'data' in doc_entry:
            if is_cache_valid(doc_entry['timestamp']):
                print(f"Using cached document for {url}")
                return doc_entry['data']
            else:
                # Remove expired entry
                del documents[url]
                cache['last_updated'] = datetime.now()
                save_document_cache(cache)
    return None

def cache_document(url, data):
    """Cache processed document"""
    cache = load_document_cache()
    # Ensure cache has the correct structure
    if not isinstance(cache, dict):
        cache = {'documents': {}, 'last_updated': datetime.now()}
    if 'documents' not in cache or not isinstance(cache['documents'], dict):
        cache['documents'] = {}
        
    cache['documents'][url] = {
        'data': data,
        'timestamp': datetime.now()
    }
    cache['last_updated'] = datetime.now()
    save_document_cache(cache)

def preprocess(text):
    """Cleans, tokenizes, removes stop words, and lemmatizes text."""
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    punct = set(string.punctuation)
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    filtered_tokens = [
        lemmatizer.lemmatize(word) for word in tokens
        if word.isalpha() and word not in stop_words and word not in punct
    ]
    return filtered_tokens

def download_and_extract_text(url):
    """Downloads a PDF from a URL and extracts its text content."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            text = "".join(page.get_text() for page in doc)
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None
    except Exception as e:
        print(f"Error processing PDF from {url}: {e}")
        return None

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple text chunking."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def make_langchain_compatible(data):
    """Convert existing data format to work with LangChain"""
    # Add any necessary format conversions here
    data['langchain_compatible'] = True
    return data

def process_new_document(document_url):
    """Process a new document URL for immediate use"""
    print(f"Processing new document: {document_url}")
    
    # Check cache first
    cached_data = get_cached_document(document_url)
    if cached_data:
        print(f"Loaded processed document from cache: {document_url}")
        return cached_data
    
    # Download and extract text
    text = download_and_extract_text(document_url)
    if not text:
        raise ValueError(f"Failed to extract text from document: {document_url}")
    
    # Create document structure
    documents = [{'id': document_url, 'text': text}]
    
    # Chunk the document
    chunks = chunk_text(text)
    chunked_documents = []
    for i, chunk_text_content in enumerate(chunks):
        chunked_documents.append({
            'chunk_id': i,
            'source_doc_id': document_url,
            'text': chunk_text_content
        })
    
    # Process for TF-IDF
    raw_texts = [doc['text'] for doc in chunked_documents]
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(raw_texts)
    
    data_to_return = {
        "full_documents": documents,
        "chunked_documents": chunked_documents,
        "vectorizer": vectorizer,
        "tfidf_matrix_chunks": tfidf_matrix
    }
    
    # Add LangChain compatibility flag
    data_to_return = make_langchain_compatible(data_to_return)
    
    # Cache the processed document
    cache_document(document_url, data_to_return)
    
    print(f"Processing complete for new document.")
    return data_to_return

def initialize_and_preprocess(document_url=None):
    """Process either a new document or load existing cache"""
    if document_url:
        # Process new document dynamically
        return process_new_document(document_url)
    
    # Existing cache loading logic (fallback)
    if os.path.exists(PERSISTENCE_FILE):
        print("Loading pre-processed data from disk...")
        with open(PERSISTENCE_FILE, 'rb') as f:
            data = pickle.load(f)
        # Ensure backwards compatibility
        if 'langchain_compatible' not in data:
            print("Updating data format for LangChain compatibility...")
            data = make_langchain_compatible(data)
            # Optionally, save the updated format back to disk
            with open(PERSISTENCE_FILE, 'wb') as f:
                pickle.dump(data, f)
        return data

    raise ValueError("No document URL provided and no cached data available.")