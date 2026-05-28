"""
CourseWise Retrieval Evaluation (Three-Way Comparison)
=================================================================================
Measures retrieval quality for three pipeline modes:
  - Baseline: vector-only search
  - Hybrid: vector + BM25 + RRF (no reranker)
  - Hybrid+Rerank: vector + BM25 + RRF + reranker

Metrics:
  - Recall@k: did the correct document appear in the top k results?
  - MRR: mean reciprocal rank of the correct document

Output: retrieval_eval_results.json

Usage: python retrieval_eval.py
"""

import json
import os
import sys
import time
import warnings

os.environ["ANONYMIZED_TELEMETRY"] = "False"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from coursewise_chatbot import initialize_chatbot, CONFIG
    from hybrid_search import reciprocal_rank_fusion
    from preprocessing import get_text
except ImportError as e:
    print(f"[ERROR] Could not import project modules: {e}")
    sys.exit(1)

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("[WARN] rank-bm25 not installed.")

try:
    from reranker import Reranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    print("[WARN] reranker not found.")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
TOP_K = 5  # Standard IR evaluation uses k=5

# Manually labeled ground truth
GROUND_TRUTH = {
    "Q01": "qa_0043",
    "Q02": "qa_0028",
    "Q03": "qa_0115",
    "Q04": "qa_0115",
    "Q05": "qa_0072",
    "Q06": "qa_0073",
    "Q07": "qa_0008",
    "Q08": "qa_0108",
    "Q09": "qa_0218",
    "Q10": "qa_0014",
    "Q11": "qa_0000",
    "Q12": "qa_0225",
    "Q13": "qa_0316",
    "Q14": "qa_0272",
    "Q15": "qa_0006",
    "Q16": "qa_0024",
    "Q17": "qa_0006",
    "Q18": "qa_0370",
    "Q19": "qa_0010",
    "Q20": "qa_0011",
    "Q21": "qa_0014",
    "Q22": "qa_0014",
    "Q23": "qa_0150",
    "Q24": "qa_0014",
    "Q25": "qa_0014",
}


TEST_QUERIES = [
    {"id": "Q01", "category": "Access & Onboarding",
     "query": "How do I get access to the ATAIN platform?"},
    {"id": "Q02", "category": "Access & Onboarding",
     "query": "What steps are needed to onboard a new institution to CourseWise?"},
    {"id": "Q03", "category": "Access & Onboarding",
     "query": "My invitation link to CourseWise has expired. What should I do?"},
    {"id": "Q04", "category": "Access & Onboarding",
     "query": "I cannot log into the CourseWise platform. How do I resolve this?"},
    {"id": "Q05", "category": "Data Use Agreement",
     "query": "What is the Data Use Agreement (DUA) for CourseWise?"},
    {"id": "Q06", "category": "Data Use Agreement",
     "query": "How do I sign the Data Use Agreement using DocuSign?"},
    {"id": "Q07", "category": "Data Use Agreement",
     "query": "Can the governing law clause in the DUA be changed to our state?"},
    {"id": "Q08", "category": "Data Use Agreement",
     "query": "We need to add redlines to the Data Use Agreement. What is the process?"},
    {"id": "Q09", "category": "Credit Articulation",
     "query": "What is credit articulation and how does CourseWise support it?"},
    {"id": "Q10", "category": "Credit Articulation",
     "query": "How does the native articulation approval workflow work?"},
    {"id": "Q11", "category": "Credit Articulation",
     "query": "Can CourseWise handle common course numbering across systems?"},
    {"id": "Q12", "category": "Data Management",
     "query": "How do I upload a course catalog into CourseWise?"},
    {"id": "Q13", "category": "Data Management",
     "query": "What data format is required for course catalog uploads?"},
    {"id": "Q14", "category": "Data Management",
     "query": "I uploaded data but am not seeing it reflected in the platform. What should I check?"},
    {"id": "Q15", "category": "Contracting",
     "query": "What information is needed to complete the supplier registration process?"},
    {"id": "Q16", "category": "Contracting",
     "query": "Who is the authorized signatory for contracts with the CourseWise provider?"},
    {"id": "Q17", "category": "Contracting",
     "query": "Can the payment structure in the contract be negotiated?"},
    {"id": "Q18", "category": "AICMP",
     "query": "What is the AI Credit Mobility Project (AICMP)?"},
    {"id": "Q19", "category": "AICMP",
     "query": "Which system office leads are involved in the AICMP project?"},
    {"id": "Q20", "category": "AICMP",
     "query": "How often does the AICMP biweekly meeting occur and who attends?"},
    {"id": "Q21", "category": "Platform Features",
     "query": "Is there a feedback mechanism within the CourseWise platform?"},
    {"id": "Q22", "category": "Platform Features",
     "query": "When will the articulation approval workflow be available?"},
    {"id": "Q23", "category": "Platform Features",
     "query": "How do I access a specific institution's data in CourseWise?"},
    {"id": "Q24", "category": "ATAIN Community",
     "query": "What is the ATAIN Knowledge Community and how do I join?"},
    {"id": "Q25", "category": "ATAIN Community",
     "query": "Who leads the ATAIN Knowledge Community sessions?"},
]


def load_corpus(dataset_path):
    """Load corpus using CONFIG's prefer_quality setting."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prefer = set(CONFIG.get("prefer_quality", ["detailed", "moderate"]))
    docs = []
    for pair in data["qa_pairs"]:
        if pair.get("tags", {}).get("response_quality", "") not in prefer:
            continue
        text = get_text(pair)
        if len(text) < 10 or len(pair.get("answer_text", "")) < 20:
            continue
        docs.append({
            "qa_id": pair["qa_id"],
            "question": text,
            "answer": pair["answer_text"],
            "metadata": pair.get("tags", {}),
        })
    return docs


def bm25_search(query, bm25_model, doc_list, top_k=10):
    """BM25 keyword search."""
    if not bm25_model or not doc_list:
        return []
    scores = bm25_model.get_scores(query.lower().split())
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {
            "qa_id": doc_list[i]["qa_id"],
            "question": doc_list[i]["question"],
            "answer": doc_list[i]["answer"],
            "metadata": doc_list[i]["metadata"],
            "bm25_score": float(scores[i]),
            "distance": None,
        }
        for i in idx
    ]


def normalize_qa_id(qa_id):
    """Strip dedup suffix added by some vector stores, e.g. qa_0001_1 -> qa_0001."""
    parts = qa_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit() and parts[0].startswith("qa_"):
        return parts[0]
    return qa_id


def check_recall(results, ground_truth_qa_id, top_k):
    """Check if ground truth document is in top-k results and return rank."""
    retrieved_ids = [normalize_qa_id(r["qa_id"]) for r in results[:top_k]]
    if ground_truth_qa_id in retrieved_ids:
        rank = retrieved_ids.index(ground_truth_qa_id) + 1
        return True, rank
    return False, None


def main():
    print(f"\n=== CourseWise Retrieval Evaluation ===")
    print(f"Queries           : {len(TEST_QUERIES)}")
    print(f"Top-k             : {TOP_K}")
    print(f"Reranker mode     : {'ON' if RERANKER_AVAILABLE else 'OFF'}\n")

    print("Initializing chatbot components...")
    chatbot = initialize_chatbot()
    print("Ready.\n")

    corpus = load_corpus(CONFIG["dataset_path"])
    print(f"Corpus size       : {len(corpus)} docs\n")

    bm25_model = None
    if BM25_AVAILABLE:
        tokenised = [d["question"].lower().split() for d in corpus]
        bm25_model = BM25Okapi(tokenised)
        print("BM25 index built.\n")

    reranker = None
    if RERANKER_AVAILABLE:
        reranker = Reranker()
        print("Reranker loaded.\n")

    print(f"Running three-way comparison...\n")

    results = []
    baseline_hits = baseline_mrr_sum = 0.0
    hybrid_hits = hybrid_mrr_sum = 0.0
    hybrid_rerank_hits = hybrid_rerank_mrr_sum = 0.0

    for q in TEST_QUERIES:
        gt_qa_id = GROUND_TRUTH[q["id"]]

        # ── Mode 1: Baseline (vector-only) ─────────────────────────────────
        baseline_results = chatbot.vector_store.search(query=q["query"], top_k=TOP_K)

        # ── Mode 2 & 3: Hybrid ─────────────────────────────────────────────
        vector_wide = chatbot.vector_store.search(query=q["query"], top_k=TOP_K * 3)

        # Perform hybrid fusion (vector + BM25 via RRF)
        if BM25_AVAILABLE and bm25_model:
            bm25_results = bm25_search(q["query"], bm25_model, corpus, top_k=TOP_K * 3)
            for r in vector_wide:
                r["qa_id"] = normalize_qa_id(r["qa_id"])
            hybrid_fused = reciprocal_rank_fusion(vector_wide, bm25_results)
        else:
            hybrid_fused = vector_wide

        # Mode 2: Hybrid without reranker (RRF only)
        hybrid_results = hybrid_fused[:TOP_K]

        # Mode 3: Hybrid with reranker
        hybrid_rerank_results = hybrid_results  # fallback
        if reranker is not None and hybrid_fused:
            for r in hybrid_fused:
                if not r.get("answer"):
                    doc = next((d for d in corpus if d["qa_id"] == r.get("qa_id")), None)
                    if doc:
                        r["question"] = r.get("question") or doc.get("question", "")
                        r["answer"] = doc.get("answer", "")
                    else:
                        r.setdefault("question", "")
                        r.setdefault("answer", "")
            try:
                hybrid_rerank_results = reranker.rerank(
                    q["query"], 
                    hybrid_fused, 
                    top_k=TOP_K
                )
            except Exception as e:
                print(f"  [WARN] reranker failed for {q['id']}: {e}")
                hybrid_rerank_results = hybrid_fused[:TOP_K]

        # ── Score all three modes ──────────────────────────────────────────
        b_hit, b_rank = check_recall(baseline_results, gt_qa_id, TOP_K)
        h_hit, h_rank = check_recall(hybrid_results, gt_qa_id, TOP_K)
        hr_hit, hr_rank = check_recall(hybrid_rerank_results, gt_qa_id, TOP_K)

        baseline_hits += int(b_hit)
        hybrid_hits += int(h_hit)
        hybrid_rerank_hits += int(hr_hit)

        baseline_mrr_sum += (1.0 / b_rank) if b_rank else 0.0
        hybrid_mrr_sum += (1.0 / h_rank) if h_rank else 0.0
        hybrid_rerank_mrr_sum += (1.0 / hr_rank) if hr_rank else 0.0

        b_label = f"HIT(r={b_rank})" if b_hit else "MISS"
        h_label = f"HIT(r={h_rank})" if h_hit else "MISS"
        hr_label = f"HIT(r={hr_rank})" if hr_hit else "MISS"

        print(f"  {q['id']}: base={b_label:<11} hybrid={h_label:<11} rerank_c5={hr_label:<11} gt={gt_qa_id}")

        results.append({
            "query_id": q["id"],
            "category": q["category"],
            "query": q["query"],
            "ground_truth_qa_id": gt_qa_id,
            "baseline": {
                "hit": b_hit,
                "rank": b_rank,
                "retrieved_ids": [normalize_qa_id(r["qa_id"]) for r in baseline_results],
            },
            "hybrid": {
                "hit": h_hit,
                "rank": h_rank,
                "search_mode": "vector+BM25+RRF (no reranker)",
                "retrieved_ids": [normalize_qa_id(r["qa_id"]) for r in hybrid_results],
            },
            "hybrid_rerank_constrained": {
                "hit": hr_hit,
                "rank": hr_rank,
                "search_mode": "vector+BM25+RRF+rerank(pool_size=5)",
                "retrieved_ids": [normalize_qa_id(r["qa_id"]) for r in hybrid_rerank_results],
            },
        })

    # ── Summary ───────────────────────────────────────────────────────────
    n = len(TEST_QUERIES)
    baseline_recall = baseline_hits / n
    hybrid_recall = hybrid_hits / n
    hybrid_rerank_recall = hybrid_rerank_hits / n

    baseline_mrr = baseline_mrr_sum / n
    hybrid_mrr = hybrid_mrr_sum / n
    hybrid_rerank_mrr = hybrid_rerank_mrr_sum / n

    print(f"\n=== Summary (k={TOP_K}) ===")
    print(f"{'Mode':<30} {'Recall':<10} {'MRR':<10}")
    print(f"{'-'*50}")
    print(f"{'Baseline (vector)':<30} {baseline_recall:.1%}      {baseline_mrr:.3f}")
    print(f"{'Hybrid (no reranker)':<30} {hybrid_recall:.1%}      {hybrid_mrr:.3f}")
    print(f"{'Hybrid + Rerank':<30} {hybrid_rerank_recall:.1%}      {hybrid_rerank_mrr:.3f}")

    print(f"\n{'Improvements vs Baseline':<30}")
    print(f"{'Hybrid alone':<30} {(hybrid_recall - baseline_recall):+.1%} recall, {(hybrid_mrr - baseline_mrr):+.3f} MRR")
    print(f"{'Hybrid+Rerank':<30} {(hybrid_rerank_recall - baseline_recall):+.1%} recall, {(hybrid_rerank_mrr - baseline_mrr):+.3f} MRR")
    print(f"{'Reranker impact (pool=5)':<30} {(hybrid_rerank_recall - hybrid_recall):+.1%} recall, {(hybrid_rerank_mrr - hybrid_mrr):+.3f} MRR")

    # ── Per-category ──────────────────────────────────────────────────────
    categories = sorted(set(q["category"] for q in TEST_QUERIES))
    print(f"\n=== Per-Category Recall@{TOP_K} ===")
    print(f"{'Category':<28} {'Baseline':<12} {'Hybrid':<12} {'Rerank':<12}")
    print(f"{'-'*64}")

    category_stats = {}
    for cat in categories:
        cat_rows = [r for r in results if r["category"] == cat]
        cat_n = len(cat_rows)
        cat_b = sum(1 for r in cat_rows if r["baseline"]["hit"])
        cat_h = sum(1 for r in cat_rows if r["hybrid"]["hit"])
        cat_hr = sum(1 for r in cat_rows if r["hybrid_rerank_constrained"]["hit"])

        print(f"{cat:<28} {cat_b/cat_n:.0%} ({cat_b}/{cat_n})   {cat_h/cat_n:.0%} ({cat_h}/{cat_n})   {cat_hr/cat_n:.0%} ({cat_hr}/{cat_n})")

        category_stats[cat] = {
            "baseline_recall": round(cat_b / cat_n, 3),
            "baseline_hits": cat_b,
            "hybrid_recall": round(cat_h / cat_n, 3),
            "hybrid_hits": cat_h,
            "hybrid_rerank_recall": round(cat_hr / cat_n, 3),
            "hybrid_rerank_hits": cat_hr,
            "n_queries": cat_n,
        }

    # ── Save JSON ─────────────────────────────────────────────────────────
    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": {
            "baseline_recall": round(baseline_recall, 3),
            "baseline_mrr": round(baseline_mrr, 3),
            "hybrid_recall": round(hybrid_recall, 3),
            "hybrid_mrr": round(hybrid_mrr, 3),
            "hybrid_rerank_recall": round(hybrid_rerank_recall, 3),
            "hybrid_rerank_mrr": round(hybrid_rerank_mrr, 3),
            "hybrid_vs_baseline_recall": round(hybrid_recall - baseline_recall, 3),
            "hybrid_vs_baseline_mrr": round(hybrid_mrr - baseline_mrr, 3),
            "reranker_pool5_impact_recall": round(hybrid_rerank_recall - hybrid_recall, 3),
            "reranker_pool5_impact_mrr": round(hybrid_rerank_mrr - hybrid_mrr, 3),
            "total_queries": n,
            "top_k": TOP_K,
        },
        "category_breakdown": category_stats,
        "per_query": results,
    }

    with open("retrieval_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: retrieval_eval_results.json")


if __name__ == "__main__":
    main()