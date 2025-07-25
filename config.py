# File: config.py
"""
Configuration settings and constants for the CAG system.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Data & Cache ---
PERSISTENCE_FILE = "processed_data.pkl" # Stores processed text, vectorizers, etc.
CACHE_FILE = "cag_cache.pkl"           # Stores the pre-computed KV caches (conceptual for HF)

# --- Add the URLs to your documents here ---
PDF_URLS = [
    "https://hackrx.in/policies/BAJHLIP23020V012223.pdf",
    "https://hackrx.in/policies/HDFHLIP23024V072223.pdf",
    "https://hackrx.in/policies/ICIHLIP22012V012223.pdf",
    "https://hackrx.in/policies/STARHLIP23020V012223.pdf",
    "https://hackrx.in/policies/ADITYBIRLAHLACT23003V022223.pdf",
    # Add more PDF URLs as needed
]

# --- Model Configuration ---
# Using the new Gemini model
LLM_MODEL_NAME = "gemini-2.5-flash" # Updated model name

# --- CAG Specific ---
# Maximum length for chunks fed to the LLM for cache generation
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100

# --- Optional: Cache Retrieval Strategy ---
# For simplicity, this example might use TF-IDF on original text to find relevant chunks
# for cache loading, then use their pre-computed caches.
RETRIEVAL_METHOD = "tfidf" # Placeholder for future expansion

# --- Gemini API Key (Loaded from .env) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables (.env file).")