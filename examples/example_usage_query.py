# examples/example_usage_query.py

import os
import argparse
from fastrag import LLM
from fastrag.querying import FastRAG

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Query FastRAG with a specific RAG strategy.")
parser.add_argument("-q", "--question", required=True, help="Question to ask the RAG system.")
parser.add_argument("-r", "--rag-type", required=True, choices=["graph", "text", "hybrid", "combined"],
                    help="RAG type to use: graph, text, hybrid, or combined.")
args = parser.parse_args()

question = args.question
rag_type = args.rag_type


# Set Up LLM
API_KEY = os.environ.get("API_KEY")
llm = LLM(model_name="gpt-4o", api_key=API_KEY, temperature=0.1)

# Initialize FastRAG
fast_rag = FastRAG(url="bolt://localhost:7687", username="neo4j", password="12345-password", llm=llm.model)
print("Schema loaded:", fast_rag.schema)

# Build RAG chains
rag_chains = {
    "graph": fast_rag.build_graph_chain(verbose=False),
    "text": fast_rag.build_text_chain(verbose=False),
    "hybrid": fast_rag.build_hybrid_chain(verbose=False),
    "combined": fast_rag.build_combined_chain(verbose=False),
}

# Run query
print(f"\nUsing '{rag_type}' RAG to answer:\n{question}\n")
answer = rag_chains[rag_type].invoke(question)
print("Answer:", answer)

# Clean up
fast_rag.close()
