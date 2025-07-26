# File: main.py
"""
Main entry point for the CAG application.
"""
import os
from cag_engine import CAGEngine
from cache_builder import build_cache, CACHE_FILE

def main():
    print("--- Cache-Augmented Generation (CAG) System ---")

    # Check if cache exists, build if not
    if not os.path.exists(CACHE_FILE):
        print(f"Cache file '{CACHE_FILE}' not found.")
        build_choice = input("Do you want to build the cache now? (y/n): ").lower()
        if build_choice == 'y':
            build_cache()
        else:
            print("Cannot run CAG system without cache. Exiting.")
            return

    # Initialize the integrated CAG Engine
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
    print("Commands: 'exit' to quit, 'report' for cache analytics, 'feedback <score>' for learning")
    
    while True:
        user_input = input("\nAsk a question or enter command: ").strip()
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        elif user_input.lower() == 'report':
            report = cag_engine.get_cache_report()
            print("\n--- Cache Performance Report ---")
            print(report)
            continue
        elif user_input.lower().startswith('feedback'):
            parts = user_input.split()
            if len(parts) == 2:
                try:
                    score = float(parts[1])
                    # Use last query for feedback (you'd need to store this)
                    print(f"Learning from feedback score: {score}")
                except ValueError:
                    print("Invalid feedback score. Use: feedback <0.0-1.0>")
            continue

        if not user_input:
            print("Please enter a valid question.")
            continue

        print("\nProcessing query...")
        answer = cag_engine.generate_answer(user_input)

        print("\n--- AI Assistant (CAG) ---")
        print(answer)
        print("-" * 50)

if __name__ == "__main__":
    main()
