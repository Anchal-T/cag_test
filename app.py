from quart import Quart, request, jsonify
from cag_engine import CAGEngine
import asyncio
import functools
from dotenv import load_dotenv
import os
import functools

load_dotenv()
app = Quart(__name__)

cag_engine = CAGEngine()

def validate_bearer_token(f):  
    @functools.wraps(f)
    async def wrapper(*args, **kwargs): 
        auth_header = request.headers.get('Authorization')
        expected_token = f"Bearer {os.getenv('BEARER_TOKEN')}"
        
        if not auth_header or auth_header != expected_token:
            return jsonify({'error': 'Unauthorized'}), 401
        
        return await f(*args, **kwargs)
    return wrapper

@app.route('/hackrx/run', methods=['POST'])
# @validate_bearer_token
async def get_answers():
    """
    API endpoint to process questions against a given document URL.
    Handles requests asynchronously for improved performance.
    """
    try:
        data = await request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        document_url = data.get('documents')
        questions = data.get('questions')
        
        if not document_url:
            return jsonify({"error": "Document URL ('documents') is required"}), 400
        
        if not questions or not isinstance(questions, list):
            return jsonify({"error": "A list of questions ('questions') is required"}), 400
        
        # Await the asynchronous batch generation function
        answers_list = await cag_engine.generate_batch_answers(questions, document_url)
        
        # Format the response
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
        print(f"Unhandled error in /hackrx/run: {e}")
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to confirm the server is running."""
    return jsonify({"status": "healthy"}), 200

def start_app():
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)

if __name__ == '__main__':
    start_app()