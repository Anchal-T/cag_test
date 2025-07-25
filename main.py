# File: main.py
from data_processor import initialize_and_preprocess
from retriever import hybrid_search
from llm_interface import configure_llm, get_llm_response
from rank_bm25 import BM25Okapi
import os

def main():
    # --- IMPORTANT: Delete the old pickle file if it exists with the wrong structure ---
    if os.path.exists('rag_data.pkl'):
        print("Checking data file structure...")
        import pickle
        with open('rag_data.pkl', 'rb') as f:
            data = pickle.load(f)
            if 'sample_papers' in data:
                print("Old data structure found. Deleting 'rag_data.pkl' to regenerate.")
                os.remove('rag_data.pkl')

    # 1. Initialize models and data from PDFs
    processed_data = initialize_and_preprocess()

    # This check makes the code robust to different data versions
    documents_list = processed_data.get('documents') or processed_data.get('sample_papers')
    if not documents_list:
        raise ValueError("Could not find 'documents' or 'sample_papers' key in processed data.")

    bm25_model = BM25Okapi(processed_data['tokenized_corpus'])
    llm_model = configure_llm()
    
    print("\nSystem is ready. You can now ask questions.")

    while True:
        query = input("\nAsk a question about your documents (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
            
        # 2. Retrieve relevant documents
        print("\nSearching for relevant documents...")
        relevant_docs = hybrid_search(
            query=query,
            documents=documents_list,
            bm25_model=bm25_model,
            vectorizer=processed_data['vectorizer'],
            tfidf_matrix=processed_data['tfidf_matrix']
        )
        
        if not relevant_docs:
            print("Could not find any relevant documents.")
            continue

        print(f"\nFound {len(relevant_docs)} relevant document(s). Retrieving context:")
        for doc in relevant_docs:
            # Displaying the URL as the ID
            print(f"  - ID: {doc['doc']['id']}, Score: {doc['score']:.4f}")

        # 3. Generate a response
        print("\nGenerating answer with LLM...")
        answer = get_llm_response(llm_model, query, relevant_docs)
        
        print("\n--- AI Assistant ---")
        print(answer)
        print("--------------------\n")

if __name__ == "__main__":
    main()
