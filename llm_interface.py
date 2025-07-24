from google import genai
import os
from dotenv import load_dotenv

def configure_llm():
    """Configures the Gemini API."""
    load_dotenv()
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-2.5-flash')
        return model
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        return None

def get_llm_response(model, query, context_docs):
    """Formats a prompt and gets a response from the Gemini API."""
    if not model:
        return "LLM service is not available."

    context = "\n\n---\n\n".join([doc['doc']['text'] for doc in context_docs])
    
    prompt = f"""
    You are a helpful AI assistant for answering questions based on research papers.
    Based *only* on the following context, please provide a concise answer to the user's question.
    If the context does not contain the answer, state that you couldn't find relevant information.

    Context:
    {context}

    Question: {query}

    Answer:
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while communicating with the LLM: {e}"        