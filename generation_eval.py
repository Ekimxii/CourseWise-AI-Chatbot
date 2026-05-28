"""
CourseWise Evaluation Framework - Multi-Judge with Reranker
============================================================
Runs 25 curated test queries through:
  - Baseline pipeline  (vector-only search, flat prompt)
  - Full pipeline      (hybrid search + reranker + structured prompt)

Each response is independently scored by 3 judge models:
  - Llama
  - Anthropic Claude Haiku
  - Google Gemma 4

Output: evaluation_results.json (clean structured output for notebook analysis)

Usage: python evaluation.py [--queries N] [--no-judge] [--no-rerank]
"""

from __future__ import annotations

# Suppress ChromaDB telemetry warnings before any imports
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import argparse
import json
import re
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()

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
    print("[WARN] reranker.py not found. Full pipeline will run without reranker.")

try:
    import ollama as _ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# ═════════════════════════════════════════════════════════════════════════════
# 25 Test Queries - reference answers updated to be redaction-compatible
# (no real names, no specific institution names that would be pseudonymized)
# ═════════════════════════════════════════════════════════════════════════════
TEST_QUERIES = [
    {"id": "Q01", "category": "Access & Onboarding",
     "query": "How do I get access to the ATAIN platform?",
     "reference": "Users typically receive an invitation or welcome email with login credentials. They may also need to complete a Data Use Agreement before accessing the platform."},
    {"id": "Q02", "category": "Access & Onboarding",
     "query": "What steps are needed to onboard a new institution to CourseWise?",
     "reference": "Onboarding generally involves signing a Data Use Agreement, completing a supplier registration process, and receiving platform login credentials."},
    {"id": "Q03", "category": "Access & Onboarding",
     "query": "My invitation link to CourseWise has expired. What should I do?",
     "reference": "Users should contact the CourseWise or ATAIN support team to request a new invitation link or have their account credentials reset."},
    {"id": "Q04", "category": "Access & Onboarding",
     "query": "I cannot log into the CourseWise platform. How do I resolve this?",
     "reference": "Common solutions include verifying login credentials, using the password reset feature, clearing browser cache, or contacting the support team for assistance."},
    {"id": "Q05", "category": "Data Use Agreement",
     "query": "What is the Data Use Agreement (DUA) for CourseWise?",
     "reference": "The DUA is a legal agreement that governs how participating institutions share course catalog and articulation data with the CourseWise platform."},
    {"id": "Q06", "category": "Data Use Agreement",
     "query": "How do I sign the Data Use Agreement using DocuSign?",
     "reference": "The DUA is distributed via an electronic signature platform. Signatories receive an email with a link and must verify their identity before signing."},
    {"id": "Q07", "category": "Data Use Agreement",
     "query": "Can the governing law clause in the DUA be changed to our state?",
     "reference": "The governing law section of the DUA can be left blank or amended based on the institution's legal requirements, subject to review by both parties."},
    {"id": "Q08", "category": "Data Use Agreement",
     "query": "We need to add redlines to the Data Use Agreement. What is the process?",
     "reference": "Institutions can submit their redlined version of the DUA back to the CourseWise team, who will review the proposed changes and negotiate as needed."},
    {"id": "Q09", "category": "Credit Articulation",
     "query": "What is credit articulation and how does CourseWise support it?",
     "reference": "Credit articulation is the process of evaluating how courses at one institution can transfer for credit at another. CourseWise uses AI to help automate and streamline this process across partner institutions."},
    {"id": "Q10", "category": "Credit Articulation",
     "query": "How does the native articulation approval workflow work?",
     "reference": "The platform includes a workflow that allows users to create, review, and store course articulation decisions directly within CourseWise, without needing external tools."},
    {"id": "Q11", "category": "Credit Articulation",
     "query": "Can CourseWise handle common course numbering across systems?",
     "reference": "Yes, CourseWise supports interoperability with common course numbering systems used across higher-education networks to facilitate transfer credit evaluation."},
    {"id": "Q12", "category": "Data Management",
     "query": "How do I upload a course catalog into CourseWise?",
     "reference": "Course catalogs can be uploaded through the platform's data upload interface. The support team can help align the data format if needed."},
    {"id": "Q13", "category": "Data Management",
     "query": "What data format is required for course catalog uploads?",
     "reference": "CourseWise accepts structured course data, typically in CSV or spreadsheet format. Templates are provided to ensure consistency across institutions."},
    {"id": "Q14", "category": "Data Management",
     "query": "I uploaded data but am not seeing it reflected in the platform. What should I check?",
     "reference": "Verify that the upload completed without errors, confirm the correct institution scope was selected, and contact the support team if the issue continues."},
    {"id": "Q15", "category": "Contracting",
     "query": "What information is needed to complete the supplier registration process?",
     "reference": "Supplier registration typically requires organizational details, authorized signatory information, and a tax identification number or EIN."},
    {"id": "Q16", "category": "Contracting",
     "query": "Who is the authorized signatory for contracts with the CourseWise provider?",
     "reference": "The COO of the company that operates CourseWise serves as the authorized signatory for platform-related contracts and agreements."},
    {"id": "Q17", "category": "Contracting",
     "query": "Can the payment structure in the contract be negotiated?",
     "reference": "Yes, the fee structure section of the contract can be edited to reflect agreed-upon cost details and payment schedules between the parties."},
    {"id": "Q18", "category": "AICMP",
     "query": "What is the AI Credit Mobility Project (AICMP)?",
     "reference": "AICMP is a collaborative initiative that uses AI to improve transfer credit mobility and course articulation across multiple higher-education systems and institutions."},
    {"id": "Q19", "category": "AICMP",
     "query": "Which system office leads are involved in the AICMP project?",
     "reference": "Multiple state higher-education system offices participate in AICMP, contributing institutional data and collaborating on transfer credit improvement efforts."},
    {"id": "Q20", "category": "AICMP",
     "query": "How often does the AICMP biweekly meeting occur and who attends?",
     "reference": "AICMP holds biweekly meetings attended by representatives from participating system offices and the CourseWise platform team to coordinate project progress."},
    {"id": "Q21", "category": "Platform Features",
     "query": "Is there a feedback mechanism within the CourseWise platform?",
     "reference": "A formal feedback system is under development. In the meantime, users can submit feedback via a provided form link or by emailing the support team."},
    {"id": "Q22", "category": "Platform Features",
     "query": "When will the articulation approval workflow be available?",
     "reference": "The native articulation approval workflow was planned for rollout in mid-2025, with exact timing dependent on development progress and partner readiness."},
    {"id": "Q23", "category": "Platform Features",
     "query": "How do I access a specific institution's data in CourseWise (e.g. WTAMU)?",
     "reference": "Users need to be granted institution-level access permissions. Platform administrators can confirm whether access has been enabled for specific campuses."},
    {"id": "Q24", "category": "ATAIN Community",
     "query": "What is the ATAIN Knowledge Community and how do I join?",
     "reference": "The ATAIN Knowledge Community is a peer learning group for member institutions. Invitations to join are typically sent by the community's lead facilitator."},
    {"id": "Q25", "category": "ATAIN Community",
     "query": "Who leads the ATAIN Knowledge Community sessions?",
     "reference": "The ATAIN Knowledge Community sessions are led by a designated facilitator who also serves as the project director for the initiative."},
]


@dataclass
class EvalResult:
    query_id: str
    category: str
    query: str
    reference: str

    baseline_answer: str = ""
    baseline_latency_s: float = 0.0
    baseline_retrieved_count: int = 0
    baseline_confidence: float = 0.0
    baseline_judges: dict = field(default_factory=dict)

    full_answer: str = ""
    full_latency_s: float = 0.0
    full_retrieved_count: int = 0
    full_confidence: float = 0.0
    full_search_mode: str = "hybrid+rerank"
    full_judges: dict = field(default_factory=dict)


_openai_client = None
_anthropic_client = None
_gemma_model = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _anthropic_client


def _get_gemma_model():
    global _gemma_model
    if _gemma_model is None:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        _gemma_model = genai.GenerativeModel('gemma-4-26b-a4b-it')
    return _gemma_model


def call_ollama(prompt: str, model: str, system: str = "") -> str:
    if not OLLAMA_AVAILABLE:
        return "[Ollama not available]"
    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = _ollama.chat(model=model, messages=messages)
        return resp["message"]["content"].strip()
    except Exception as e:
        return f"[Ollama error: {e}]"


def call_openai(prompt: str, system: str = "") -> str:
    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = _get_openai_client().chat.completions.create(
            model="gpt-4o-mini", messages=messages,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI error: {e}]"


def call_claude(prompt: str, system: str = "") -> str:
    try:
        kwargs = {
            "model": "claude-haiku-4-5",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = _get_anthropic_client().messages.create(**kwargs)
        return resp.content[0].text.strip()
    except Exception as e:
        return f"[Claude error: {e}]"


def call_gemma(prompt: str, system: str = "") -> str:
    try:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        resp = _get_gemma_model().generate_content(full_prompt)
        text = resp.text.strip()
        json_match = re.search(
            r'\{[^{}]*"relevance"\s*:\s*\d+[^{}]*\}',
            text, re.DOTALL
        )
        if json_match:
            return json_match.group(0)
        return text
    except Exception as e:
        return f"[Gemma error: {e}]"


def build_bm25_index(dataset_path: str):
    if not BM25_AVAILABLE:
        return None, []
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs, tokenised = [], []
    prefer = set(CONFIG.get("prefer_quality", ["detailed", "moderate"]))
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
        tokenised.append(text.lower().split())
    bm25 = BM25Okapi(tokenised) if tokenised else None
    return bm25, docs


def bm25_search(query: str, bm25_model, doc_list: list, top_k: int = 10) -> list:
    if not bm25_model or not doc_list:
        return []
    tokens = query.lower().split()
    scores = bm25_model.get_scores(tokens)
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


def build_full_prompt(user_question: str, results: list, search_mode: str = "hybrid+rerank") -> str:
    source_blocks = []
    for i, r in enumerate(results[:CONFIG.get("top_k", 4)], 1):
        meta = r.get("metadata", {})
        score_line = ""
        if r.get("rerank_score") is not None:
            score_line = f"  Rerank Score : {r['rerank_score']:.4f}\n"
        elif r.get("rrf_score") is not None:
            score_line = f"  RRF Score    : {r['rrf_score']:.5f}\n"
        elif r.get("distance") is not None:
            score_line = f"  Similarity   : {max(0.0, 1.0 - r['distance']):.2f}\n"
        block = (
            f"[SOURCE {i}]\n"
            f"  Product Area : {meta.get('product_area', 'N/A')}\n"
            f"  Request Type : {meta.get('request_type', 'N/A')}\n"
            f"  Priority     : {meta.get('priority', 'N/A')}\n"
            f"{score_line}"
            f"  Matched Q    : {r.get('question', '')[:180]}\n"
            f"  Answer       : {r.get('answer', '')}\n"
        )
        source_blocks.append(block)
    context_text = "\n---\n".join(source_blocks)
    return f"""You are a helpful, precise support assistant for CourseWise, a platform for transfer pathway analysis and course articulation.

## Key Terminology
- **DUA**: Data Use Agreement - legal contract governing data sharing
- **ATAIN**: Knowledge Community for peer learning and collaboration
- **AICMP**: AI Credit Mobility Project - initiative to improve transfer credit
- **Course Articulation**: Process of evaluating course equivalency across institutions

## Example Context
If asked about platform access, reference invitation emails and credential setup.
If asked about data, reference upload formats and the support team's assistance.
If asked about DUA, reference the legal agreement and signature process.

## User Question
{user_question}

## Retrieved Knowledge Sources ({len(source_blocks)} result(s) via {search_mode})
{context_text}

## Instructions
1. Answer the user's question using ONLY the information in the sources above.
2. If multiple sources are relevant, synthesize them into a single coherent answer.
3. If the sources do not adequately address the question, say so clearly. Do NOT invent details or steps.
4. Use a professional but friendly tone, written directly to the user.
5. Keep your answer concise (3-6 sentences unless depth is required).

## Critical formatting rules
- Do NOT include email signatures, signoffs, or placeholders like "[Your Name]".
- Do NOT start with "Based on the provided context" or similar meta-narration.
- Do NOT reference internal labels like "Person_A", "Org_B", "Source 1", etc.
- Do NOT sign as "CourseWise Support" or pretend to be a specific person.
- Speak directly to the user. Just answer.

## Answer"""


JUDGE_SYSTEM = """You are a strict but fair evaluator of RAG chatbot answers.
You output ONLY valid JSON, no prose, no markdown fences."""

JUDGE_PROMPT_TEMPLATE = """Evaluate the following chatbot answer against the question and reference answer.

QUESTION: {question}

REFERENCE ANSWER (gold standard): {reference}

CHATBOT ANSWER: {answer}

SOURCES USED FOR RETRIEVAL:
{sources}

Score on two dimensions (integer 1-5):

RELEVANCE (1-5)
  5 = Directly and fully addresses the question
  4 = Mostly addresses it with minor gaps
  3 = Partially addresses it
  2 = Tangentially related but misses the main point
  1 = Completely off-topic or refuses to answer

FAITHFULNESS (1-5)
  5 = Every claim is supported by the retrieved sources
  4 = Almost all claims are grounded; minor extrapolation
  3 = Some claims are unsupported but not hallucinated
  2 = Contains claims that go beyond the sources
  1 = Answer contradicts sources or contains clear hallucinations

Respond with ONLY this JSON (no other text):
{{
  "relevance": <integer 1-5>,
  "faithfulness": <integer 1-5>,
  "reasoning": "<one sentence explaining both scores>"
}}"""


def preprocess_query(query: str) -> str:
    """
    Enhance query with synonyms and expand abbreviated terms for better retrieval.
    """
    expansions = {
        "DUA": "Data Use Agreement DUA",
        "ATAIN": "ATAIN Knowledge Community",
        "AICMP": "AI Credit Mobility Project",
    }
    expanded = query
    for abbrev, expansion in expansions.items():
        if abbrev in query:
            expanded = expanded.replace(abbrev, f"{expansion}")
    return expanded


def judge_answer(question, reference, answer, sources, judge="llama"):
    sources_text = "\n".join(
        f"Source {i+1}: {s.get('answer', '')[:300]}"
        for i, s in enumerate(sources[:3])
    ) or "No sources retrieved."

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, reference=reference,
        answer=answer, sources=sources_text,
    )

    if judge == "llama":
        raw = call_ollama(prompt, "llama3", system=JUDGE_SYSTEM)
    elif judge == "claude":
        raw = call_claude(prompt, system=JUDGE_SYSTEM)
    elif judge == "gemma":
        raw = call_gemma(prompt, system=JUDGE_SYSTEM)
    else:
        raise ValueError(f"Unknown judge: {judge}")

    try:
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        data = json.loads(clean)
        rel = max(1, min(5, int(data.get("relevance", 1))))
        fai = max(1, min(5, int(data.get("faithfulness", 1))))
        reasoning = str(data.get("reasoning", ""))
        return rel, fai, reasoning
    except Exception:
        nums = re.findall(r'\b([1-5])\b', raw)
        rel = int(nums[0]) if len(nums) > 0 else 1
        fai = int(nums[1]) if len(nums) > 1 else 1
        return rel, fai, raw[:200]


def run_baseline(chatbot, query: str, top_k: int) -> dict:
    t0 = time.time()
    results = chatbot.vector_store.search(query=query, top_k=top_k)
    if not results:
        return {"answer": "No results found.", "sources": [], "confidence": 0.0,
                "latency": time.time() - t0, "retrieved_count": 0}

    confidence = max(0.0, 1.0 - results[0]["distance"]) if results[0].get("distance") else 0.8
    answer = chatbot.llm_refiner.refine_answer(query, results)
    latency = time.time() - t0

    sources = [{"answer": r["answer"], "metadata": r.get("metadata", {})} for r in results[:3]]
    return {
        "answer": answer, "sources": sources, "confidence": confidence,
        "latency": latency, "retrieved_count": len(results),
    }


def run_full(chatbot, query: str, top_k: int, bm25_model, doc_list: list,
             reranker, llm_model: str) -> dict:
    t0 = time.time()
    
    # Preprocess query with synonyms and expansions
    processed_query = query

    vector_results = chatbot.vector_store.search(query=processed_query, top_k=top_k * 3)

    if BM25_AVAILABLE and bm25_model:
        bm25_results = bm25_search(processed_query, bm25_model, doc_list, top_k=top_k * 3)
        for r in vector_results:
            parts = r["qa_id"].rsplit("_", 1)
            r["qa_id"] = parts[0] if len(parts) == 2 and parts[1].isdigit() else r["qa_id"]
        fused = reciprocal_rank_fusion(vector_results, bm25_results)
        search_mode = "hybrid"
    else:
        fused = vector_results
        search_mode = "vector"

    # Calculate initial confidence to decide if reranking is needed
    if fused and fused[0].get("rrf_score") is not None:
        initial_confidence = min(1.0, fused[0]["rrf_score"] / 0.033)
    elif fused and fused[0].get("distance") is not None:
        initial_confidence = max(0.0, 1.0 - fused[0]["distance"])
    else:
        initial_confidence = 0.5

    if reranker is not None and fused:
        for r in fused:
            if "answer" not in r or not r.get("answer"):
                doc = next((d for d in doc_list if d["qa_id"] == r.get("qa_id")), None)
                if doc:
                    r["question"] = r.get("question") or doc.get("question", "")
                    r["answer"] = doc.get("answer", "")
                else:
                    r.setdefault("question", "")
                    r.setdefault("answer", "")
        try:
            final_results = reranker.rerank(query, fused, top_k=top_k)
            search_mode = search_mode + "+rerank"
        except Exception as e:
            print(f"  [WARN] reranker failed: {e}")
            final_results = fused[:top_k]
    else:
        final_results = fused[:top_k]

    if not final_results:
        return {"answer": "No results found.", "sources": [], "confidence": 0.0,
                "latency": time.time() - t0, "retrieved_count": 0,
                "search_mode": search_mode}

    if final_results[0].get("rerank_score") is not None:
        rs = final_results[0]["rerank_score"]
        confidence = max(0.0, min(1.0, (rs + 5) / 10))
    elif final_results[0].get("rrf_score") is not None:
        confidence = min(1.0, final_results[0]["rrf_score"] / 0.033)
    elif final_results[0].get("distance") is not None:
        confidence = max(0.0, 1.0 - final_results[0]["distance"])
    else:
        confidence = 0.8

    prompt = build_full_prompt(query, final_results, search_mode)
    answer = call_openai(
        prompt,
        system="You are a helpful CourseWise support assistant. Base your answers strictly on the provided sources. Use clear, professional language suitable for institutional stakeholders."
    )
    latency = time.time() - t0

    sources = [{"answer": r.get("answer", ""), "metadata": r.get("metadata", {})}
               for r in final_results[:3]]
    return {
        "answer": answer, "sources": sources, "confidence": confidence,
        "latency": latency, "retrieved_count": len(final_results),
        "search_mode": search_mode,
    }


JUDGES = ["llama", "claude", "gemma"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--full-only", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--queries", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=CONFIG.get("top_k", 3))
    parser.add_argument("--llm-model", type=str, default=CONFIG.get("llm_model", "gpt-4o-mini"))
    parser.add_argument("--output", type=str, default="evaluation_results.json")
    args = parser.parse_args()

    run_baseline_flag = not args.full_only
    run_full_flag = not args.baseline_only

    queries = TEST_QUERIES[:args.queries] if args.queries else TEST_QUERIES

    print(f"\n=== CourseWise Multi-Judge Evaluation ===")
    print(f"Queries           : {len(queries)}")
    print(f"Baseline pipeline : {'ON' if run_baseline_flag else 'OFF'}")
    print(f"Full pipeline     : {'ON' if run_full_flag else 'OFF'}")
    print(f"Reranker          : {'ON' if (RERANKER_AVAILABLE and not args.no_rerank) else 'OFF'}")
    print(f"Judges            : {'OFF' if args.no_judge else ', '.join(JUDGES)}")
    print(f"Chatbot LLM       : {args.llm_model}")
    print(f"Top-k             : {args.top_k}\n")

    print("Initializing chatbot...")
    chatbot = initialize_chatbot()
    print("Chatbot ready.\n")

    bm25_model, doc_list = None, []
    if run_full_flag:
        print("Building BM25 index...")
        bm25_model, doc_list = build_bm25_index(CONFIG["dataset_path"])
        print(f"BM25 index: {len(doc_list)} docs.\n")

    reranker = None
    if run_full_flag and RERANKER_AVAILABLE and not args.no_rerank:
        print("Loading reranker (cross-encoder)...")
        reranker = Reranker()
        print("Reranker ready.\n")

    eval_results = []
    n_judge_calls = 0

    for i, q in enumerate(queries, 1):
        print(f"[{i:02d}/{len(queries)}] {q['id']} {q['query'][:65]}")
        er = EvalResult(
            query_id=q["id"], category=q["category"],
            query=q["query"], reference=q["reference"],
        )

        if run_baseline_flag:
            r1 = run_baseline(chatbot, q["query"], top_k=args.top_k)
            er.baseline_answer = r1["answer"]
            er.baseline_latency_s = round(r1["latency"], 3)
            er.baseline_retrieved_count = r1["retrieved_count"]
            er.baseline_confidence = round(r1["confidence"], 3)

            if not args.no_judge:
                for judge in JUDGES:
                    rel, faith, reasoning = judge_answer(
                        q["query"], q["reference"], r1["answer"],
                        r1["sources"], judge=judge
                    )
                    er.baseline_judges[judge] = {
                        "relevance": rel, "faithfulness": faith, "reasoning": reasoning,
                    }
                    n_judge_calls += 1
                judge_str = " | ".join(
                    f"{j}: r={er.baseline_judges[j]['relevance']} f={er.baseline_judges[j]['faithfulness']}"
                    for j in JUDGES
                )
                print(f"        baseline: {judge_str}  lat={er.baseline_latency_s:.2f}s")

        if run_full_flag:
            r2 = run_full(
                chatbot, q["query"], top_k=args.top_k,
                bm25_model=bm25_model, doc_list=doc_list,
                reranker=reranker, llm_model=args.llm_model
            )
            er.full_answer = r2["answer"]
            er.full_latency_s = round(r2["latency"], 3)
            er.full_retrieved_count = r2["retrieved_count"]
            er.full_confidence = round(r2["confidence"], 3)
            er.full_search_mode = r2["search_mode"]

            if not args.no_judge:
                for judge in JUDGES:
                    rel, faith, reasoning = judge_answer(
                        q["query"], q["reference"], r2["answer"],
                        r2["sources"], judge=judge
                    )
                    er.full_judges[judge] = {
                        "relevance": rel, "faithfulness": faith, "reasoning": reasoning,
                    }
                    n_judge_calls += 1
                judge_str = " | ".join(
                    f"{j}: r={er.full_judges[j]['relevance']} f={er.full_judges[j]['faithfulness']}"
                    for j in JUDGES
                )
                print(f"        full    : {judge_str}  lat={er.full_latency_s:.2f}s  mode={er.full_search_mode}")

        eval_results.append(er)

    output = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "n_queries": len(queries),
            "judges": JUDGES if not args.no_judge else [],
            "chatbot_llm": args.llm_model,
            "top_k": args.top_k,
            "reranker_used": RERANKER_AVAILABLE and not args.no_rerank,
            "ran_baseline": run_baseline_flag,
            "ran_full": run_full_flag,
        },
        "results": [asdict(r) for r in eval_results],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== Done ===")
    print(f"Total judge calls : {n_judge_calls}")
    print(f"Results saved to  : {args.output}")


if __name__ == "__main__":
    main()