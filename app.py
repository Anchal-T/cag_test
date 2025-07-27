from flask import Flask, request, jsonify
from cag_engine import CAGEngine
from cache_builder import build_cache, CACHE_FILE
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global CAG engine instance
cag_engine = None

def initialize_cag_engine():
    """Initialize the CAG engine"""
    global cag_engine
    
    # Initialize the integrated CAG Engine without a specific document
    try:
        cag_engine = CAGEngine()
        logger.info("CAG Engine initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing CAG Engine: {e}")
        raise

@app.route('/hackrx/run', methods=['POST'])
def get_answers():
    """
    API endpoint to get answers for questions based on document URL
    Expected JSON input: 
    {
        "documents": "document_url", 
        "questions": ["question1", "question2", ...]
    }
    """
    global cag_engine
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        document_url = data.get('documents')
        questions = data.get('questions')
        
        if not document_url:
            return jsonify({"error": "Document URL is required"}), 400
            
        if not questions or not isinstance(questions, list):
            return jsonify({"error": "Questions array is required"}), 400
        
        # Process questions using CAG engine with the provided document
        if cag_engine is None:
            return jsonify({"error": "CAG engine not initialized"}), 500
            
        # Use batch processing for better performance
        answers_list = cag_engine.generate_batch_answers(questions, document_url)
        
        # Format responses
        answers = []
        for i, question in enumerate(questions):
            answers.append({
                "question": question,
                "answer": answers_list[i] if i < len(answers_list) else "Error: No response generated"
            })
        
        return jsonify({
            "document_url": document_url,
            "answers": answers
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "cag_engine": cag_engine is not None}), 200

if __name__ == '__main__':
    # Initialize CAG engine when starting the server
    try:
        initialize_cag_engine()
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")