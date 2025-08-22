import sys
from crew_setup import agentic_rag_answer

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Provide a query.")
        sys.exit(1)

    query = sys.argv[1]
    print("🚀 Running Agentic RAG pipeline...\n")
    result = agentic_rag_answer(query)
    print("\n✅ Final Output:\n", result)
