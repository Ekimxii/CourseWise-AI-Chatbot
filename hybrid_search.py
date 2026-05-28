from preprocessing import get_text

def reciprocal_rank_fusion(vector_results: list, bm25_results: list, k: int = 60) -> list:
    """
    Combine vector and BM25 ranked lists into one merged ranking.
    
    Args:
        vector_results: list of dicts with 'qa_id' key, ordered by vector rank
        bm25_results: list of dicts with 'qa_id' key, ordered by BM25 rank
        k: constant to prevent top results from dominating (default 60)
    
    Returns:
        merged list of dicts sorted by RRF score descending
    """
    scores = {}
    all_results = {}
    
    # Score vector results
    for rank, result in enumerate(vector_results):
        qa_id = result['qa_id']
        scores[qa_id] = scores.get(qa_id, 0) + 1 / (rank + k)
        all_results[qa_id] = result
    
    # Score BM25 results
    for rank, result in enumerate(bm25_results):
        qa_id = result['qa_id']
        scores[qa_id] = scores.get(qa_id, 0) + 1 / (rank + k)
        all_results[qa_id] = result
    
    # Sort by combined score
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    
    return [
        {**all_results[qa_id], 'rrf_score': scores[qa_id]}
        for qa_id in sorted_ids
    ]


if __name__ == "__main__":
    # Scenario: user asks "how do I reset my CourseWise password"
    
    # Vector search found semantically similar results about account access
    vector_results = [
        {"qa_id": "A", "question": "I cannot log into the platform"},
        {"qa_id": "B", "question": "How do I get access to CourseWise?"},
        {"qa_id": "C", "question": "My invitation link expired"},
        {"qa_id": "D", "question": "How do I reset my CourseWise password"},
    ]
    
    # BM25 found exact keyword matches for "reset" and "password"
    bm25_results = [
        {"qa_id": "D", "question": "How do I reset my CourseWise password"},
        {"qa_id": "E", "question": "Password reset link not working"},
        {"qa_id": "A", "question": "I cannot log into the platform"},
        {"qa_id": "F", "question": "Reset my account credentials"},
    ]
    
    merged = reciprocal_rank_fusion(vector_results, bm25_results)
    
    print("VECTOR SEARCH alone (semantic):")
    for r in vector_results:
        print(f"  {r['qa_id']}: {r['question']}")
    
    print("\nBM25 SEARCH alone (keyword):")
    for r in bm25_results:
        print(f"  {r['qa_id']}: {r['question']}")
    
    print("\nRRF MERGED (best of both):")
    for r in merged:
        print(f"  {r['qa_id']}: {r['rrf_score']:.4f} - {r['question']}")