import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

class QueryProcessor:
    def __init__(self):
        # Load spaCy model for NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
            
    def enhance_query(self, query):
        """Enhance query with synonyms and related terms"""
        enhanced_terms = [query]
        
        if self.nlp:
            doc = self.nlp(query)
            
            # Extract entities and important terms
            entities = [ent.text for ent in doc.ents]
            enhanced_terms.extend(entities)
            
            # Add root words and lemmas
            important_tokens = [token.lemma_ for token in doc 
                              if not token.is_stop and token.pos_ in ['NOUN', 'VERB', 'ADJ']]
            enhanced_terms.extend(important_tokens)
            
        return list(set(enhanced_terms))  # Remove duplicates
        
    def detect_query_intent(self, query):
        """Detect the intent of the query for better routing"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['coverage', 'covered', 'include']):
            return 'coverage_inquiry'
        elif any(word in query_lower for word in ['claim', 'file', 'submit']):
            return 'claim_process'
        elif any(word in query_lower for word in ['premium', 'cost', 'price']):
            return 'pricing_inquiry'
        elif any(word in query_lower for word in ['exclusion', 'not covered', 'exclude']):
            return 'exclusion_inquiry'
        else:
            return 'general_inquiry'