import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import pickle
import os
import requests
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from config import PERSISTENCE_FILE, PDF_URLS
from tqdm import tqdm

# --- Download NLTK data (only need to do this once) ---
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

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

def download_and_extract_text(url):
    """Downloads a PDF from a URL and extracts its text content."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Use PyMuPDF (fitz) to open the PDF from memory
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            text = "".join(page.get_text() for page in doc)
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None
    except Exception as e:
        print(f"Error processing PDF from {url}: {e}")
        return None

def initialize_and_preprocess():
    """
    Loads data from persistence file if it exists. 
    Otherwise, downloads and processes PDFs, then saves them.
    """
    if os.path.exists(PERSISTENCE_FILE):
        print("Loading pre-processed data from disk...")
        with open(PERSISTENCE_FILE, 'rb') as f:
            data = pickle.load(f)
        return data

    print("No pre-processed data found. Starting one-time processing from PDF URLs...")
    
    documents = []
    for url in tqdm(PDF_URLS, desc="Downloading & Parsing PDFs"):
        text = download_and_extract_text(url)
        if text:
            documents.append({'id': url, 'text': text})

    if not documents:
        raise ValueError("No documents could be processed. Check URLs and network connection.")

    print("Processing documents for BM25...")
    tokenized_corpus = [preprocess(doc['text']) for doc in documents]

    print("Processing documents for TF-IDF...")
    raw_texts = [doc['text'] for doc in documents]
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, min_df=2) # min_df ignores rare terms
    tfidf_matrix = vectorizer.fit_transform(raw_texts)
    
    data_to_persist = {
        "documents": documents,
        "tokenized_corpus": tokenized_corpus,
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix
    }
    
    print(f"Processing complete. Saving to {PERSISTENCE_FILE}...")
    with open(PERSISTENCE_FILE, 'wb') as f:
        pickle.dump(data_to_persist, f)
        
    return data_to_persist
