import sys
import json
import logging
from typing import List, Dict, Any
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.config import settings
from src.graph import build_rag_graph
from src.ingestion import run_ingestion
from src.utils import ensure_knowledge_base_pdf

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_test_suite")


BENCHMARK_QUERIES = [
    {
        "id": 1,
        "category": "Definition & Scope",
        "query": "What is the core definition of Agentic AI as outlined in the eBook?",
        "expect_grounded": True
    },
    {
        "id": 2,
        "category": "Architecture & Paradigms",
        "query": "What are the main architectural components required to build agentic systems?",
        "expect_grounded": True
    },
    {
        "id": 3,
        "category": "Use Cases",
        "query": "What real-world industry use cases for Agentic AI are discussed in the eBook?",
        "expect_grounded": True
    },
    {
        "id": 4,
        "category": "Comparison",
        "query": "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
        "expect_grounded": True
    },
    {
        "id": 5,
        "category": "Challenges & Considerations",
        "query": "What key challenges or limitations of Agentic AI are mentioned in the document?",
        "expect_grounded": True
    },
    {
        "id": 6,
        "category": "Out-of-Scope Test (Groundedness Check)",
        "query": "What is the capital of France?",
        "expect_grounded": False
    }
]


def run_benchmark_tests():
    """Executes verification suite across all 6 benchmark test queries."""
    print("=" * 80)
    print("STARTING INDUSTRIAL BENCHMARK EVALUATION FOR LANGGRAPH & PINECONE RAG CHATBOT")
    print("=" * 80)

    # 1. Ensure Knowledge Base Document
    ensure_knowledge_base_pdf()

    # 2. Run ETL Ingestion Pipeline
    print("\n[Step 1] Running ETL Ingestion & Vector Storage Pipeline...")
    run_ingestion()

    # 3. Build Compiled LangGraph Workflow
    print("\n[Step 2] Compiling LangGraph State Machine...")
    graph = build_rag_graph(index_name=settings.PINECONE_INDEX_NAME)

    results: List[Dict[str, Any]] = []
    passed_count = 0

    print("\n[Step 3] Executing Benchmark Queries...")
    print("-" * 80)

    for item in BENCHMARK_QUERIES:
        q_id = item["id"]
        category = item["category"]
        query_text = item["query"]
        expect_grounded = item["expect_grounded"]

        print(f"\nTest #{q_id} [{category}]")
        print(f"Query: '{query_text}'")

        initial_state = {
            "question": query_text,
            "context": [],
            "answer": "",
            "score": 0.0,
            "is_grounded": False
        }

        # Invoke LangGraph workflow
        state_output = graph.invoke(initial_state)

        formatted_payload = {
            "query": query_text,
            "final_answer": state_output.get("answer", ""),
            "retrieved_context_chunks": state_output.get("context", []),
            "confidence_score": state_output.get("score", 0.0)
        }

        # Verification check logic
        score = formatted_payload["confidence_score"]
        answer = formatted_payload["final_answer"]

        if expect_grounded:
            is_pass = (score > 0.3) and ("cannot answer" not in answer.lower())
        else:
            # Out of scope query should refuse or return 0.0 score
            is_pass = (score == 0.0) or ("cannot answer" in answer.lower()) or ("not available" in answer.lower())

        if is_pass:
            passed_count += 1
            status_str = "[PASS]"
        else:
            status_str = "[FAIL]"

        print(f"Status: {status_str}")
        print(f"Confidence Score: {score}")
        print(f"Final Answer: {answer}")
        print(f"Retrieved Chunks Count: {len(formatted_payload['retrieved_context_chunks'])}")
        print("JSON Payload:")
        print(json.dumps(formatted_payload, indent=2))
        print("-" * 80)

        results.append({
            "id": q_id,
            "category": category,
            "status": status_str,
            "score": score,
            "pass": is_pass
        })

    # Summary Dashboard
    print("\n" + "=" * 80)
    print("BENCHMARK TEST SUITE EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Benchmark Tests: {len(BENCHMARK_QUERIES)}")
    print(f"Passed: {passed_count} / {len(BENCHMARK_QUERIES)}")
    print(f"Success Rate: {(passed_count / len(BENCHMARK_QUERIES)) * 100:.1f}%\n")

    for res in results:
        print(f"Test #{res['id']} [{res['category']}]: {res['status']} (Score: {res['score']})")

    print("=" * 80)
    
    if passed_count == len(BENCHMARK_QUERIES):
        print("ALL BENCHMARK VERIFICATION TESTS PASSED SUCCESSFULLY!")
        return 0
    else:
        print("SOME BENCHMARK TESTS DID NOT MEET EXPECTED GROUNDING THRESHOLD.")
        return 1



if __name__ == "__main__":
    exit_code = run_benchmark_tests()
    sys.exit(exit_code)
