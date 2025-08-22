import os

# LLM Model (Prodigy ADK wraps this)
LLM_MODEL = "gpt-4"  

# Vector DB Settings
CHROMA_DIR = "chroma_db"
CHROMA_GLOBAL_COLLECTION_NAME = "knowledge_base"

# Embeddings model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# System Instructions for Agents
SYSTEM_INSTRUCTIONS = {
    "retriever": "Retrieve the most relevant JSON or PDF text chunks.",
    "planner": "Analyze the query + context, create structured step-by-step execution plan.",
    "developer": "Generate complete, production-grade code grounded in retrieved JSON and plan.",
}
