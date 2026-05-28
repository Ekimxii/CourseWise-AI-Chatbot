"""
Cross-encoder reranker for the full pipeline.
Reranks the full hybrid-search candidate pool by semantic relevance to the query,
then returns the top_k results.
"""
from sentence_transformers import CrossEncoder
from typing import List, Dict


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        Rerank the full candidate pool using cross-encoder semantic relevance,
        then return the top_k results.

        Args:
            query:   The search query
            results: List of result dicts with 'question' and 'answer' fields
            top_k:   Number of final results to return

        Returns:
            List of top_k reranked results
        """
        if not results:
            return []

        pairs = [
            (query, f"{r.get('question', '')} {r.get('answer', '')}")
            for r in results
        ]
        scores = self.model.predict(pairs)

        reranked = [
            {**r, "rerank_score": float(score)}
            for r, score in zip(results, scores)
        ]
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_k]