import os
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig

def configure_llm():
    """Initializes the Gemini client."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY or GOOGLE_API_KEY not set")
        return None

    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        return None

def get_llm_response(client, query, context_docs):
    """Generates a response based on context documents."""
    if not client:
        return "LLM service is not available."

    context = "\n\n---\n\n".join(doc['doc']['text'] for doc in context_docs)
    prompt = (
        "You are a helpful AI assistant answering user questions from research papers.\n"
        "Use *only* the provided context. If there's no answer in the context, state that clearly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=GenerateContentConfig(thinking_config=ThinkingConfig(thinking_budget=0))
        )
        return resp.text
    except Exception as e:
        return f"Error from Gemini API: {e}"
