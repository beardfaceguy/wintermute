import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.rag_utils import init_rag, is_rag_valid, query_rag

print("Checking index validity...")
if not is_rag_valid():
    print("Index invalid or missing. Initializing...")
    init_rag()
else:
    print("Index is valid.")

question = "What did the monkey do?"
print(f"Query: {question}")
response = query_rag(question)
print(f"Response:\n{response}")
