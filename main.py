# File: main.py
from tokenize_words import initialize_and_preprocess
from retriever import hybrid_search
from llm_interface import configure_llm, get_llm_response
from rank_bm25 import BM25Okapi

def main():
    # 1. Initialize models and data
    processed_data = initialize_and_preprocess()
    bm25_model = BM25Okapi(processed_data['tokenized_corpus'])
    llm_model = configure_llm()
    
    print("\nSystem is ready. You can now ask questions.")

    while True:
        query = input("\nAsk a question about AI papers (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
            
        # 2. Retrieve relevant documents
        print("\nSearching for relevant documents...")
        relevant_docs = hybrid_search(
            query=query,
            sample_papers=processed_data['sample_papers'],
            bm25_model=bm25_model,
            vectorizer=processed_data['vectorizer'],
            tfidf_matrix=processed_data['tfidf_matrix']
        )
        
        if not relevant_docs:
            print("Could not find any relevant documents.")
            continue

        print(f"\nFound {len(relevant_docs)} relevant document(s). Retrieving context:")
        for doc in relevant_docs:
            print(f"  - ID: {doc['doc']['name']}, Score: {doc['score']:.4f}")

        # 3. Generate a response
        print("\nGenerating answer with LLM...")
        answer = get_llm_response(llm_model, query, relevant_docs)
        
        print("\n--- AI Assistant ---")
        print(answer)
        print("--------------------\n")

if __name__ == "__main__":
    main()
