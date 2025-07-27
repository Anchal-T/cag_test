# File: llm_interface.py

import asyncio
from google import genai
from google.genai.types import GenerateContentConfig

from config import LLM_MODEL_NAME, GEMINI_API_KEY

# Configure synchronous client
sync_client = genai.Client(api_key=GEMINI_API_KEY)

# Configure asynchronous client (aio namespace)
async_client = genai.Client(api_key=GEMINI_API_KEY)

def get_llm_response_with_cache(query, relevant_cache_entries):
    """Blocking, synchronous generation using GenAI SDK."""
    if not relevant_cache_entries:
        return "No relevant knowledge found for the query."

    cached_text = "\n---\n".join(e.get("text_snippet", "N/A") for e in relevant_cache_entries)
    prompt = f"""
You are a helpful AI assistant that answers questions based on provided knowledge.
Pre‑loaded knowledge:
{cached_text}

Question: {query}
Answer:
"""

    try:
        resp = sync_client.models.generate_content(
            model=LLM_MODEL_NAME,
            contents=prompt,
            config=GenerateContentConfig(max_output_tokens=500, temperature=0.2)
        )
        return resp.text.strip()
    except Exception as e:
        return f"Error during generation: {e}"

async def get_llm_response_async(query, relevant_entries):
    """Non-blocking, async generation suitable for parallel requests."""
    if not relevant_entries:
        return "No relevant knowledge found for the query."

    cached_text = "\n---\n".join(e.get("text_snippet", "N/A") for e in relevant_entries)
    prompt = f"""
You are a helpful AI assistant that answers questions based on provided knowledge.
Pre‑loaded knowledge:
{cached_text}

Question: {query}
Answer:
"""

    try:
        resp = await async_client.aio.models.generate_content(
            model=LLM_MODEL_NAME,
            contents=prompt,
            config=GenerateContentConfig(max_output_tokens=500, temperature=0.2)
        )
        return resp.text.strip()
    except Exception as e:
        # Use logging or print as appropriate in production
        print(f"Async generation error for query '{query}': {e}")
        return f"Error during generation: {e}"

# Example function to issue many async calls in parallel:
async def fetch_responses_in_parallel(queries_and_contexts):
    tasks = [
        get_llm_response_async(query, entries)
        for query, entries in queries_and_contexts
    ]
    return await asyncio.gather(*tasks)

# Optionally wrap the parallel calls in blocking context:
def fetch_parallel_sync(queries_and_contexts):
    return asyncio.run(fetch_responses_in_parallel(queries_and_contexts))
