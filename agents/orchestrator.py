# agents/orchestrator.py
"""
Orchestrateur Doxa - Version Complète
Coordonne tous les agents : Triage → Query → Retrieval → Evaluator → Response
"""

import time
import uuid
from agents.triage_agent import TriageAgent
from agents.query_processor import SmartQueryProcessor
from agents.retrieval_agent import RetrievalAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.response_composer import ResponseComposer


class OrchestratorAgent:
    """
    Orchestrateur principal du système Doxa Support AI
    Pipeline complet : Triage → Query → Retrieval → Evaluation → Response
    """
    
    def __init__(self, document_processor):
        """
        Initialise l'orchestrateur avec tous les agents
        
        Args:
            document_processor: Instance de DocumentProcessor avec index FAISS chargé
        """
        
        print("🔧 Initialisation de l'Orchestrator complet...\n")
        
        # Agents de traitement
        print("   • Chargement Triage Agent...")
        self.triage = TriageAgent()
        
        print("   • Chargement Query Processor...")
        self.query_processor = SmartQueryProcessor()
        
        print("   • Chargement Retrieval Agent...")
        self.retrieval = RetrievalAgent(document_processor)
        
        print("   • Chargement Evaluator Agent...")
        self.evaluator = EvaluatorAgent()
        
        print("   • Chargement Response Composer...")
        self.response_composer = ResponseComposer()
        
        print("\n✅ Orchestrator prêt !")
        print(f"   📊 Index FAISS : {document_processor.index.ntotal} vecteurs")
        print(f"   📚 Documents : {len(set(m['source'] for m in document_processor.metadata))}")
        print(f"   ✂️  Chunks : {len(document_processor.documents)}\n")
    
    def process_ticket(self, ticket_id: str, question: str) -> dict:
        """
        Traite un ticket client de bout en bout
        
        Pipeline :
        1. Triage (validation + exceptions)
        2. Query Processing (analyse + classification)
        3. Retrieval (RAG)
        4. Evaluation (décision traiter/escalader)
        5. Response (génération réponse structurée)
        
        Args:
            ticket_id: ID unique du ticket
            question: Question du client
        
        Returns:
            dict avec status, résultats, temps d'exécution
        """
        
        # Générer trace_id pour logging
        trace_id = str(uuid.uuid4())[:8]
        
        # Démarrer le chronomètre
        start_time = time.time()
        
        print("=" * 70)
        print(f"🎫 TICKET #{ticket_id} [trace:{trace_id}]")
        print("=" * 70)
        print(f"❓ Question : {question}\n")
        
        # ══════════════════════════════════════════════════════════
        # ÉTAPE 1 : Triage (validation + détection exceptions)
        # ══════════════════════════════════════════════════════════
        print("🔍 ÉTAPE 1/5 : Triage...")
        triage_start = time.time()
        
        triage_result = self.triage.analyze(question)
        
        triage_time = time.time() - triage_start
        
        print(f"   ⏱️  Temps : {triage_time:.2f}s")
        print(f"   Cohérent     : {'✅' if triage_result['coherent'] else '❌'}")
        print(f"   Type         : {triage_result['type_question']}")
        
        if triage_result['exceptions']:
            print(f"   ⚠️  Exceptions   : {', '.join(triage_result['exceptions'])}")
        
        print()
        
        # Gestion des cas d'escalade immédiate
        if not triage_result['coherent']:
            total_time = time.time() - start_time
            return {
                "ticket_id": ticket_id,
                "question": question,
                "status": "rejected",
                "message": "Question incohérente",
                "triage": triage_result,
                "execution_time": {
                    "total": f"{total_time:.2f}s",
                    "triage": f"{triage_time:.2f}s"
                },
                "trace_id": trace_id
            }
        
        if 'emotion_negative' in triage_result['exceptions']:
            total_time = time.time() - start_time
            return {
                "ticket_id": ticket_id,
                "question": question,
                "status": "escalated",
                "reason": "Émotion négative détectée (client en colère)",
                "priority": "HAUTE",
                "triage": triage_result,
                "execution_time": {
                    "total": f"{total_time:.2f}s",
                    "triage": f"{triage_time:.2f}s"
                },
                "trace_id": trace_id
            }
        
        if 'data_sensitive' in triage_result['exceptions']:
            total_time = time.time() - start_time
            return {
                "ticket_id": ticket_id,
                "question": question,
                "status": "escalated",
                "reason": "Données sensibles détectées (mot de passe, carte...)",
                "priority": "CRITIQUE",
                "triage": triage_result,
                "execution_time": {
                    "total": f"{total_time:.2f}s",
                    "triage": f"{triage_time:.2f}s"
                },
                "trace_id": trace_id
            }
        
        # ══════════════════════════════════════════════════════════
        # ÉTAPE 2 : Query Processing (analyse + classification)
        # ══════════════════════════════════════════════════════════
        print("🧠 ÉTAPE 2/5 : Analyse et classification...")
        query_start = time.time()
        
        query_data = self.query_processor.process(question, triage_result)
        
        query_time = time.time() - query_start
        
        print(f"   ⏱️  Temps : {query_time:.2f}s")
        print(f"   Résumé       : {query_data['resume'][:60]}...")
        print(f"   Catégorie    : {query_data['categorie']}")
        print(f"   Documents    : {', '.join(query_data['documents'])}")
        print(f"   Confiance    : {query_data['confidence']:.0%}\n")
        
        # ══════════════════════════════════════════════════════════
        # ÉTAPE 3 : Retrieval (recherche dans index FAISS)
        # ══════════════════════════════════════════════════════════
        print("🔍 ÉTAPE 3/5 : Recherche augmentée (RAG)...")
        retrieval_start = time.time()
        
        retrieval_result = self.retrieval.retrieve(
            query_data=query_data,
            top_k=5,
            use_augmentation=True
        )
        
        retrieval_time = time.time() - retrieval_start
        
        print(f"   ⏱️  Temps : {retrieval_time:.2f}s")
        print(f"   Chunks       : {retrieval_result['num_chunks']}")
        print(f"   Sources      : {', '.join(retrieval_result['sources'])}")
        print(f"   Score moyen  : {retrieval_result['avg_score']:.3f}\n")
        
        # ══════════════════════════════════════════════════════════
        # ÉTAPE 4 : Evaluation (décision traiter/escalader)
        # ══════════════════════════════════════════════════════════
        print("⚖️  ÉTAPE 4/5 : Évaluation de confiance...")
        eval_start = time.time()
        
        evaluation = self.evaluator.evaluate(retrieval_result, query_data)
        
        eval_time = time.time() - eval_start
        
        print(f"   ⏱️  Temps : {eval_time:.2f}s")
        print(f"   Décision     : {evaluation['decision'].upper()}")
        print(f"   Confiance    : {evaluation['confidence_finale']:.0%}")
        print(f"   Raison       : {evaluation['raison']}\n")
        
        # Si escalade nécessaire, arrêter ici
        if evaluation['decision'] == 'escalader':
            total_time = time.time() - start_time
            return {
                "ticket_id": ticket_id,
                "question": question,
                "status": "escalated",
                "reason": evaluation['raison'],
                "priority": evaluation.get('priorite_escalade', 'MOYENNE').upper(),
                "triage": triage_result,
                "query_data": query_data,
                "retrieval": retrieval_result,
                "evaluation": evaluation,
                "execution_time": {
                    "total": f"{total_time:.2f}s",
                    "triage": f"{triage_time:.2f}s",
                    "query_processing": f"{query_time:.2f}s",
                    "retrieval": f"{retrieval_time:.2f}s",
                    "evaluation": f"{eval_time:.2f}s"
                },
                "trace_id": trace_id
            }
        
        # ══════════════════════════════════════════════════════════
        # ÉTAPE 5 : Response (génération réponse structurée)
        # ══════════════════════════════════════════════════════════
        print("✍️  ÉTAPE 5/5 : Génération de la réponse...")
        response_start = time.time()
        
        response_data = self.response_composer.compose(
            question=question,
            context=retrieval_result['context'],
            query_data=query_data,
            evaluation=evaluation
        )
        
        response_time = time.time() - response_start
        
        print(f"   ⏱️  Temps : {response_time:.2f}s")
        print(f"   Langue       : {response_data['langue']}")
        print(f"   Qualité      : {response_data['quality_score']:.0%}")
        print(f"   Mots         : {response_data['word_count']}\n")
        
        # ══════════════════════════════════════════════════════════
        # Temps total
        # ══════════════════════════════════════════════════════════
        total_time = time.time() - start_time
        
        print("⏱️  TEMPS D'EXÉCUTION")
        print("─" * 70)
        print(f"   Triage       : {triage_time:.2f}s")
        print(f"   Query Proc.  : {query_time:.2f}s")
        print(f"   Retrieval    : {retrieval_time:.2f}s")
        print(f"   Evaluation   : {eval_time:.2f}s")
        print(f"   Response     : {response_time:.2f}s")
        print(f"   ════════════════════════════")
        print(f"   TOTAL        : {total_time:.2f}s\n")
        
        # ══════════════════════════════════════════════════════════
        # Résultat final
        # ══════════════════════════════════════════════════════════
        return {
            "ticket_id": ticket_id,
            "question": question,
            "status": "completed",
            "triage": triage_result,
            "query_data": query_data,
            "retrieval": retrieval_result,
            "evaluation": evaluation,
            "response_data": response_data,
            "execution_time": {
                "triage": f"{triage_time:.2f}s",
                "query_processing": f"{query_time:.2f}s",
                "retrieval": f"{retrieval_time:.2f}s",
                "evaluation": f"{eval_time:.2f}s",
                "response": f"{response_time:.2f}s",
                "total": f"{total_time:.2f}s"
            },
            "trace_id": trace_id
        }