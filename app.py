from sympy.utilities.decorator import threaded
from flask import Flask, request, jsonify
from cag_engine import CAGEngine
import asyncio

app = Flask(__name__)

# Instantiate the engine when the app starts.
cag_engine = CAGEngine()

@app.route('/hackrx/run', methods=['POST'])
async def get_answers():  # Make the endpoint function async
    """
    API endpoint to process questions against a given document URL.
    Handles requests asynchronously for improved performance.
    """
    try:
        data = request.get_json()
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

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
