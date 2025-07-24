import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download necessary NLTK data (only need to do this once)
nltk.download('punkt')
nltk.download('stopwords')

def preprocess(text):
    """Tokenizes, removes stop words, and cleans text."""
    stop_words = set(stopwords.words('english'))
    punct = set(string.punctuation)
    
    # Tokenize and convert to lower case
    tokens = word_tokenize(text.lower())
    
    # Filter out stop words and punctuation
    filtered_tokens = [
        word for word in tokens 
        if word.isalpha() and word not in stop_words and word not in punct
    ]
    return filtered_tokens

# # Your sample papers
# sample_papers = [
#     {'id': 'attention', 'text': 'The dominant sequence transduction models...'},
#     # ... more papers
# ]

# Preprocess all documents
# tokenized_corpus = [preprocess(doc['text']) for doc in sample_papers]
