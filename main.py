# File: main.py
"""
Main entry point for the CAG application.
"""
import os
from cag_engine import CAGEngine
from cache_builder import build_cache, CACHE_FILE # Import build function

def main():
    print("--- Cache-Augmented Generation (CAG) System ---")

    # --- Check if cache exists, build if not ---
    if not os.path.exists(CACHE_FILE):
        print(f"Cache file '{CACHE_FILE}' not found.")
        build_choice = input("Do you want to build the cache now? (y/n): ").lower()
        if build_choice == 'y':
            build_cache()
        else:
            print("Cannot run CAG system without cache. Exiting.")
            return

    # --- Initialize the CAG Engine ---
    try:
        cag_engine = CAGEngine()
    except FileNotFoundError as e:
        print(f"Error initializing CAG Engine: {e}")
        print("Please ensure the cache is built correctly.")
        return
    except Exception as e:
        print(f"Unexpected error initializing CAG Engine: {e}")
        return

    print("\nCAG System is ready. You can now ask questions.")
    print("(Type 'exit' to quit)")
    while True:
        query = input("\nAsk a question based on the pre-loaded knowledge: ")
        if query.lower() == 'exit':
            print("Goodbye!")
            break

        if not query.strip():
            print("Please enter a valid question.")
            continue

        # --- Generate Answer using CAG Engine ---
        print("\nProcessing query...")
        answer = cag_engine.generate_answer(query)

        # --- Display Answer ---
        print("\n--- AI Assistant (CAG) ---")
        print(answer)
        print("-" * 20)

if __name__ == "__main__":
    main()
