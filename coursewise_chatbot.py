"""
CourseWise RAG Chatbot - Production Module
Retrieval-Augmented Generation chatbot for CourseWise support.
"""

import json
import os
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
import chromadb

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

CONFIG = {
    'dataset_path': os.path.join(os.path.dirname(__file__), 'qa_pairs_dataset_redacted.json'),
    'chroma_db_path': os.path.join(os.path.dirname(__file__), 'chroma_db'),
    'embedding_model': 'sentence-transformers',
    'sentence_model': 'all-MiniLM-L6-v2',
    'openai_model': 'text-embedding-3-small',
    'llm_model': 'gpt-4.1-mini',
    'top_k': 3,
    'min_similarity': 0.1,
    'use_llm_refinement': True,
    'prefer_quality': ['detailed', 'moderate'],
}


@dataclass
class QAPair:
    qa_id: str
    question: str
    answer: str
    metadata: Dict
    subject: str
    thread_id: str


class EmbeddingModel:
    def __init__(self, model_type: str = 'sentence-transformers'):
        self.model_type = model_type

        if model_type == 'openai':
            if not OPENAI_API_KEY:
                print("Warning: OPENAI_API_KEY not found. Falling back to sentence-transformers.")
                model_type = 'sentence-transformers'
            else:
                from openai import OpenAI
                self.client = OpenAI(api_key=OPENAI_API_KEY)
                self.model_name = CONFIG['openai_model']

        if model_type == 'sentence-transformers':
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(CONFIG['sentence_model'])
            self.model_name = CONFIG['sentence_model']
            print(f"Using local embedding model: {self.model_name}")

        self.model_type = model_type

    def embed(self, texts: List[str]) -> np.ndarray:
        if self.model_type == 'openai':
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            return np.array([item.embedding for item in response.data])
        else:
            return self.model.encode(texts, convert_to_numpy=True)

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class VectorStore:
    def __init__(self, db_path: str, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="coursewise_qa",
            metadata={"description": "CourseWise Q&A pairs"}
        )

    def add_qa_pairs(self, qa_pairs: List[QAPair]):
        if len(self.collection.get()['ids']) > 0:
            print("Vector store already populated. Skipping.")
            return

        print(f"Adding {len(qa_pairs)} Q&A pairs to vector store...")

        questions = [qa.question for qa in qa_pairs]
        embeddings = self.embedding_model.embed(questions)

        metadatas = []
        ids = []
        seen_ids = {}

        for qa in qa_pairs:
            metadata = {
                'qa_id': qa.qa_id,
                'answer': qa.answer,
                'product_area': qa.metadata.get('product_area', ''),
                'priority': qa.metadata.get('priority', ''),
                'request_type': qa.metadata.get('request_type', ''),
                'response_quality': qa.metadata.get('response_quality', ''),
                'subject': qa.subject,
                'thread_id': qa.thread_id,
            }
            metadatas.append(metadata)

            base_id = qa.qa_id
            if base_id in seen_ids:
                seen_ids[base_id] += 1
                unique_id = f"{base_id}_{seen_ids[base_id]}"
            else:
                seen_ids[base_id] = 0
                unique_id = base_id
            ids.append(unique_id)

        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=questions,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"Added {len(qa_pairs)} Q&A pairs to vector store.")

    def search(self, query: str, top_k: int = 3, filters: Optional[Dict] = None) -> List[Dict]:
        query_embedding = self.embedding_model.embed_single(query)

        where = {}
        if filters:
            for key in ('product_area', 'request_type', 'priority'):
                if key in filters:
                    where[key] = filters[key]

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where if where else None,
        )

        formatted = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                formatted.append({
                    'qa_id': results['ids'][0][i],
                    'question': results['documents'][0][i],
                    'answer': results['metadatas'][0][i]['answer'],
                    'metadata': {
                        'product_area': results['metadatas'][0][i].get('product_area', ''),
                        'priority': results['metadatas'][0][i].get('priority', ''),
                        'request_type': results['metadatas'][0][i].get('request_type', ''),
                        'response_quality': results['metadatas'][0][i].get('response_quality', ''),
                        'subject': results['metadatas'][0][i].get('subject', ''),
                    },
                    'distance': results['distances'][0][i] if 'distances' in results else None,
                })
        return formatted


class LLMRefiner:
    def __init__(self, model: str = 'gpt-4.1-mini'):
        self.model = model
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print(f"Using Ollama model: {self.model}")

    def refine_answer(self, user_question: str, retrieved_answers: List[Dict]) -> str:
        if not CONFIG['use_llm_refinement']:
            return retrieved_answers[0]['answer'] if retrieved_answers else "I couldn't find a specific answer."

        context = ""
        for i, result in enumerate(retrieved_answers[:3], 1):
            context += f"Answer {i}: {result['answer']}\n\n"

        prompt = f"""You are a helpful support assistant for CourseWise, a platform for transfer and articulation support.

User Question: {user_question}

Relevant answers from knowledge base:
{context}

Instructions:
1. Answer based only on the retrieved answers above.
2. Combine naturally if multiple are relevant.
3. Professional but friendly tone.
4. Do NOT invent information.
5. If answers don't fully address the question, say so.

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful CourseWise support assistant. Always base your answers on the provided context."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI refinement failed: {e}")
            return retrieved_answers[0]['answer'] if retrieved_answers else "I couldn't find a specific answer."


class CourseWiseChatbot:
    def __init__(self, vector_store: VectorStore, llm_refiner: LLMRefiner):
        self.vector_store = vector_store
        self.llm_refiner = llm_refiner

    def chat(self, user_question: str, filters: Optional[Dict] = None) -> Dict:
        if not user_question or len(user_question.strip()) < 3:
            return {
                'answer': "Please ask a question about CourseWise, transfer pathways, or credit articulation.",
                'sources': [],
                'confidence': 0.0,
            }

        retrieved_results = self.vector_store.search(
            query=user_question,
            top_k=CONFIG['top_k'],
            filters=filters,
        )

        if not retrieved_results:
            return {
                'answer': "I couldn't find a specific answer. Please contact support for assistance.",
                'sources': [],
                'confidence': 0.0,
            }

        if retrieved_results[0].get('distance') is not None:
            confidence = max(0.0, 1.0 - retrieved_results[0]['distance'])
        else:
            confidence = 0.8

        answer = self.llm_refiner.refine_answer(user_question, retrieved_results)

        sources = []
        for result in retrieved_results[:2]:
            sources.append({
                'answer': result['answer'][:200] + '...' if len(result['answer']) > 200 else result['answer'],
                'metadata': result['metadata'],
            })

        return {
            'answer': answer,
            'sources': sources,
            'confidence': round(confidence, 3),
            'retrieved_count': len(retrieved_results),
        }


def load_qa_dataset(file_path: str) -> List[QAPair]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    qa_pairs = []
    for pair in data['qa_pairs']:
        if pair.get('tags', {}).get('response_quality', '') not in CONFIG['prefer_quality']:
            continue

        subject = pair.get('subject', '')
        question_text = pair.get('question_text', '')
        combined_question = f"{subject} {question_text}".strip()

        if len(combined_question) < 10 or len(pair.get('answer_text', '')) < 20:
            continue

        qa_pairs.append(QAPair(
            qa_id=pair['qa_id'],
            question=combined_question,
            answer=pair['answer_text'],
            metadata=pair.get('tags', {}),
            subject=subject,
            thread_id=pair.get('thread_id', ''),
        ))

    return qa_pairs


def initialize_chatbot(dataset_path: Optional[str] = None, db_path: Optional[str] = None) -> CourseWiseChatbot:
    if dataset_path:
        CONFIG['dataset_path'] = dataset_path
    if db_path:
        CONFIG['chroma_db_path'] = db_path

    embedding_model = EmbeddingModel(CONFIG['embedding_model'])
    vector_store = VectorStore(CONFIG['chroma_db_path'], embedding_model)

    if len(vector_store.collection.get()['ids']) == 0:
        qa_pairs = load_qa_dataset(CONFIG['dataset_path'])
        vector_store.add_qa_pairs(qa_pairs)

    llm_refiner = LLMRefiner(CONFIG['llm_model'])
    return CourseWiseChatbot(vector_store, llm_refiner)


if __name__ == "__main__":
    print("Initializing CourseWise Chatbot...")
    chatbot = initialize_chatbot()

    test_queries = [
        "How do I access the ATAIN platform?",
        "What is credit articulation?",
    ]
    for query in test_queries:
        print(f"\nUser: {query}")
        result = chatbot.chat(query)
        print(f"Bot: {result['answer'][:200]}...")
        print(f"Confidence: {result['confidence']:.2f}")