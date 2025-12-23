# agents/retrieval_agent.py
import faiss
import numpy as np
from typing import List, Dict


class RetrievalAgent:
    """
    Agent de recherche avec augmentation de requête
    """
    
    def __init__(self, document_processor):
        self.doc_processor = document_processor
        self.embedder = document_processor.embedder
        self.index = document_processor.index
        self.documents = document_processor.documents
        self.metadata = document_processor.metadata
    
    def retrieve(
        self, 
        query_data: dict, 
        top_k: int = 5,
        use_augmentation: bool = True
    ) -> dict:
        """Recherche avec augmentation"""
        
        # Query Augmentation
        if use_augmentation:
            augmented_queries = self._augment_query(query_data)
        else:
            augmented_queries = [query_data['reformulation']]
        
        # Retrieval Multiple
        all_results = []
        for query in augmented_queries:
            results = self._search_faiss(
                query=query,
                selected_docs=query_data.get('documents', []),
                top_k=top_k * 2
            )
            all_results.extend(results)
        
        # Déduplication
        deduplicated = self._deduplicate_results(all_results)
        
        # Reranking
        reranked = self._rerank_results(
            query_data['reformulation'],
            deduplicated
        )
        
        final_results = reranked[:top_k]
        
        # Construction contexte
        context = self._build_context(final_results)
        
        return {
            "chunks": final_results,
            "context": context,
            "sources": list(set([r['source'] for r in final_results])),
            "num_chunks": len(final_results),
            "avg_score": sum(r['score'] for r in final_results) / len(final_results) if final_results else 0.0
        }
    
    def _augment_query(self, query_data: dict) -> List[str]:
        base_query = query_data['reformulation']
        keywords = query_data.get('mots_cles', [])
        category = query_data.get('categorie', '')
        
        augmented = [base_query]
        
        if keywords:
            top_keywords = ' '.join(keywords[:3])
            augmented.append(f"{base_query} {top_keywords}")
        
        category_context = {
            'technique': 'problème erreur bug',
            'authentification': 'login connexion accès',
            'projets_taches': 'projet tâche création',
            'collaboration': 'équipe partage notification',
            'integrations': 'intégration API connexion',
            'facturation': 'plan prix paiement',
            'securite': 'sécurité 2FA permission',
            'onboarding': 'démarrage configuration guide'
        }
        
        if category in category_context:
            context = category_context[category]
            augmented.append(f"{base_query} {context}")
        
        return list(dict.fromkeys(augmented))
    
    def _search_faiss(self, query: str, selected_docs: List[str], top_k: int) -> List[Dict]:
        query_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        
        scores, indices = self.index.search(query_emb, top_k * 3)
        
        filtered_results = []
        
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            
            meta = self.metadata[idx]
            
            if not selected_docs or meta['source'] in selected_docs:
                filtered_results.append({
                    "text": self.documents[idx],
                    "source": meta['source'],
                    "chunk_id": meta['chunk_id'],
                    "score": float(score),
                    "idx": int(idx)
                })
            
            if len(filtered_results) >= top_k:
                break
        
        if not filtered_results and selected_docs:
            for idx, score in zip(indices[0][:top_k], scores[0][:top_k]):
                if idx == -1:
                    continue
                meta = self.metadata[idx]
                filtered_results.append({
                    "text": self.documents[idx],
                    "source": meta['source'],
                    "chunk_id": meta['chunk_id'],
                    "score": float(score),
                    "idx": int(idx)
                })
        
        return filtered_results
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        seen = {}
        
        for result in results:
            key = f"{result['source']}_{result['chunk_id']}"
            
            if key not in seen or result['score'] > seen[key]['score']:
                seen[key] = result
        
        return sorted(seen.values(), key=lambda x: x['score'], reverse=True)
    
    def _rerank_results(self, original_query: str, results: List[Dict]) -> List[Dict]:
        if not results:
            return results
        
        query_emb = self.embedder.encode([original_query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        
        texts = [r['text'] for r in results]
        chunk_embs = self.embedder.encode(texts, convert_to_numpy=True)
        faiss.normalize_L2(chunk_embs)
        
        similarities = np.dot(chunk_embs, query_emb.T).flatten()
        
        for i, result in enumerate(results):
            result['rerank_score'] = float(similarities[i])
        
        return sorted(results, key=lambda x: x['rerank_score'], reverse=True)
    
    def _build_context(self, results: List[Dict]) -> str:
        if not results:
            return "Aucun contexte trouvé."
        
        context = ""
        
        for i, result in enumerate(results, 1):
            context += f"\n{'='*60}\n"
            context += f"[Extrait {i}] Source : {result['source']}.pdf\n"
            context += f"Score : {result.get('rerank_score', result['score']):.3f}\n"
            context += f"{'='*60}\n"
            context += result['text']
            context += "\n"
        
        return context