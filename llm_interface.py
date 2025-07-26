from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig

from config import LLM_MODEL_NAME, GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)
def get_llm_response_with_cache(query, relevant_cache_entries, context_docs_snippets=None):
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
