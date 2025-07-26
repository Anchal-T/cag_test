# File: llm_interface.py
"""
Manages communication with the Gemini 2.5 Flash or Flash‑Lite model via latest
Google Gen AI Python SDK.
"""
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig

from config import LLM_MODEL_NAME, GEMINI_API_KEY

# --- Configure the Google Gen AI client ---
# Decide whether to use direct API or Vertex AI based on environment
# For direct API:
client = genai.Client(api_key=GEMINI_API_KEY)

# If you prefer Vertex AI integration (Cloud project), use:
# client = genai.Client(
#     vertexai=True,
#     project="YOUR_CLOUD_PROJECT",
#     location="YOUR_LOCATION"
# )

def get_llm_response_with_cache(query, relevant_cache_entries, context_docs_snippets=None):
    """
    Formats the prompt using query and cache snippets, then calls the Gen AI SDK
    to generate a response. Uses the thinking mechanism for Gemini 2.5 Flash series.
    """
    if not relevant_cache_entries:
        return "No relevant knowledge found for the query."

    cached_knowledge_text = "\n---\n".join(
        entry.get("text_snippet", "N/A") for entry in relevant_cache_entries
    )

    context_text = ""
    if context_docs_snippets:
        context_text = "\n---\n".join(context_docs_snippets)

    prompt = f"""
You are a helpful AI assistant that answers questions based on provided knowledge.
Pre‑loaded knowledge:
{cached_knowledge_text}

Additional context (for reference):
{context_text}

Question: {query}
Answer:
"""

    try:
        # Choose model, e.g. "gemini-2.5-flash" or "gemini-2.5-flash-lite"
        response = client.models.generate_content(
            model=LLM_MODEL_NAME,
            contents=prompt,
            config=GenerateContentConfig(
                thinking_config=ThinkingConfig(thinking_budget=0)  # tuning budget as desired
            )
        )

        return response.text.strip()

    except Exception as e:
        return f"Error during generation: {e}"
