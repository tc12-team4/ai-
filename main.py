"""
Doxa Support AI - Système de Support Automatisé
Point d'entrée principal
"""
import json
import re
import os
import time
import faiss
import numpy as np
from dotenv import load_dotenv
from agents.document_processor import DocumentProcessor
from agents.orchestrator import OrchestratorAgent
from agno.agent import Agent
from agno.models.mistral import MistralChat
from sentence_transformers import SentenceTransformer

load_dotenv()

def extract_json(text: str) -> dict:
    clean = re.sub(r"```json|```", "", text).strip()
    return json.loads(clean)

# Initialiser l'embedder pour les escalations
embedder = SentenceTransformer('all-MiniLM-L6-v2')
documents = []
metadata = []

# Créer un index FAISS pour stocker les escalations
dimension = embedder.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(dimension)  # Index FAISS initial

mistral_model = MistralChat(
    id="mistral-small-latest",
    temperature=0.3
)

Evaluator = Agent(
    name="Evaluator",
    model=mistral_model,
    description=(
        "You are an evaluator agent.\n"
        "Input is a JSON object containing:\n"
        "- sources: list of sources\n"
        "- chunks_used: integer\n"
        "- confidence: float between 0.3 and 1\n\n"
        "Tasks:\n"
        "- If confidence < 0.6, return evaluation='escalade' and include a 'context' explaining why escalation is needed.\n"
        "- If confidence >= 0.6, return evaluation='C'est haut' and no context is needed.\n\n"
        "Return ONLY valid JSON like this:\n"
        "{\n"
        '  "evaluation": "...",\n'
        '  "context": "..."  # Only for escalade case\n'
        "}"
    ),
    tools=[],
)

StructuredResponder = Agent(
    name="StructuredResponder",
    model=mistral_model,
    description=(
        "You are a support response agent.\n\n"
        "Task:\n"
        "- Detect the language of the user's issue (French or English).\n"
        "- Respond in the SAME language as the user.\n"
        "- Produce a clear, professional, and structured response.\n"
        "- Use bullet points.\n"
        "- Keep it concise.\n\n"
        "You MUST follow this response template:\n"
        "1. Remerciements (thanks the user for the request)\n"
        "2. Problème (brief summary of the issue)\n"
        "3. Solution (proposed solution or explanation)\n"
        "4. Action (clear next steps for the user)\n\n"
        "Return ONLY the structured response without any additional text. and dont write any titles like remerciements, problème, solution or action"
    ),
    tools=[]
)

def setup_knowledge_base(pdf_dir: str, force_rebuild: bool = False):
    """
    Initialise la base documentaire (EXÉCUTÉ 1 SEULE FOIS)
    """
    print("\n" + "=" * 70)
    print("📚 SETUP : INITIALISATION BASE DOCUMENTAIRE")
    print("=" * 70 + "\n")
    
    setup_start = time.time()
    
    processor = DocumentProcessor(pdf_dir)
    
    # Essayer de charger l'index existant
    if not force_rebuild and processor.load_index():
        load_time = time.time() - setup_start
        
        print("✅ Index FAISS chargé depuis le disque")
        print(f"   • Temps de chargement : {load_time:.2f}s")
        print(f"   • Vecteurs            : {processor.index.ntotal}")
        print(f"   • Chunks              : {len(processor.documents)}")
        print(f"   • Documents           : {len(set(m['source'] for m in processor.metadata))}\n")
    else:
        print("🔨 Création de l'index FAISS (peut prendre 1-2 min)...\n")
        
        processor.process()  # PDF → chunks → embeddings → FAISS
        
        build_time = time.time() - setup_start
        
        print(f"\n✅ Index créé et sauvegardé")
        print(f"   • Temps total : {build_time:.2f}s\n")
    
    return processor

def main():
    """
    Boucle principale de traitement des tickets
    """
    print("\n" + "=" * 70)
    print("🤖 DOXA SUPPORT AI - SYSTÈME COMPLET")
    print("=" * 70 + "\n")
    
    # SETUP : Charger la base documentaire
    PDF_DIR = "./documents"
    
    if not os.path.exists(PDF_DIR):
        print(f"❌ ERREUR : Le dossier '{PDF_DIR}' n'existe pas !")
        print(f"   Créez le dossier et placez-y vos PDFs Doxa.\n")
        return
    
    doc_processor = setup_knowledge_base(PDF_DIR, force_rebuild=False)
    orchestrator = OrchestratorAgent(doc_processor)
    
    print("=" * 70)
    print("🧪 TESTS - TRAITEMENT DES TICKETS")
    print("=" * 70 + "\n")
    
    test_questions = [
        "Erreur 500 lors de la création de projet",
        "Quel est le prix du plan Pro et combien de membres ?",
        "Mon mot de passe est Doxa2025 et je ne peux pas me connecter",
        "C'EST INADMISSIBLE !!! 3ème fois que mes données sont perdues !!!",
        "Comment intégrer Doxa avec Slack pour notre équipe ?",
        "azefjkl qsdmlkfj 12345 ???"
    ]
    
    results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'─' * 70}")
        print(f"TEST {i}/{len(test_questions)}")
        print(f"{'─' * 70}\n")
        
        # Traiter le ticket
        result = orchestrator.process_ticket(
            ticket_id=f"T{i:03d}",
            question=question
        )
        
        results.append(result)
        
        print("=" * 70)
        print("📊 RÉSULTAT FINAL")
        print("=" * 70)
        
        print(f"\n🎫 Ticket : {result['ticket_id']}")
        print(f"📌 Status : {result['status'].upper()}")
        
        if result['status'] == 'retrieved':
            retrieval = result['retrieval']
            
            # Préparer l'input pour l'évaluateur
            inputs = {
                "sources": retrieval['sources'],
                "chunks_used": retrieval['num_chunks'],
                "confidence": retrieval['avg_score']
            }
            
            print(f"\n📚 Recherche :")
            print(f"   • Chunks récupérés : {retrieval['num_chunks']}")
            print(f"   • Sources          : {', '.join(retrieval['sources'])}")
            print(f"   • Score moyen      : {retrieval['avg_score']:.0%}")
            
            # Évaluation
            prompt = f"Input JSON: {json.dumps(inputs)}"
            response = Evaluator.run(prompt)
            evaluation_data = extract_json(response.content)
            
            print(f"\n📋 Évaluation : {evaluation_data['evaluation']}")
            
            if 'context' in evaluation_data:
                print(f"💬 Contexte : {evaluation_data['context']}")
                
                # Générer la réponse structurée
                structured_response = StructuredResponder.run(
                    f""" 
                    Ticket summary:
                    Cette question nécessite une intervention manuelle.

                    Context:
                    {evaluation_data['context']}

                    Question:
                    {question}
                    """
                )
                
                print("\n💡 RÉPONSE STRUCTURÉE :")
                print("-" * 50)
                print(structured_response.content)
                print("-" * 50)
            else:
                print(f"✅ Confiance suffisante ({retrieval['avg_score']:.0%}) - Pas besoin d'escalade")
        
        elif result['status'] == 'escalated':
            print(f"\n🚨 Raison   : {result['reason']}")
            print(f"🔴 Priorité : {result['priority']}")
            
            # Stocker l'escalade dans FAISS
            escalation_prompt = (
                f"Escalation required for ticket {result['ticket_id']}.\n"
                f"Priority: {result['priority']}\n"
                f"Reason: {result['reason']}\n"
                f"Suggested action: Immediate manual review by support team."
            )
            
            source = f"ticket_{result['ticket_id']}_escalation"
            
            # Ajouter au système de suivi des escalades
            embedding = embedder.encode([escalation_prompt], convert_to_numpy=True)
            index.add(embedding)
            
            documents.append(escalation_prompt)
            metadata.append({
                "source": source,
                "ticket_id": result['ticket_id'],
                "priority": result['priority'],
                "reason": result['reason']
            })
            
            print("\n📦 BASE DE DONNÉES FAISS (Escalades)")
            print("-" * 50)
            print(f"Total vecteurs escalades : {index.ntotal}")
            print(f"Dernière escalade ajoutée : {source}")
            
        elif result['status'] == 'rejected':
            print(f"\n❌ Message : {result['message']}")
        
        print(f"\n⏱️  Temps d'exécution : {result['execution_time']['total']}")
        print("=" * 70 + "\n")
        
        input("⏸️  Appuyez sur Entrée pour continuer...\n")
    
    # Statistiques finales
    print("\n" + "=" * 70)
    print("📈 STATISTIQUES FINALES")
    print("=" * 70)
    
    total = len(results)
    rejected = sum(1 for r in results if r['status'] == 'rejected')
    escalated = sum(1 for r in results if r['status'] == 'escalated')
    retrieved = sum(1 for r in results if r['status'] == 'retrieved')
    
    print(f"\nTotal tickets     : {total}")
    print(f"Rejetés          : {rejected} ({rejected/total*100:.0f}%)")
    print(f"Escaladés        : {escalated} ({escalated/total*100:.0f}%)")
    print(f"Traités          : {retrieved} ({retrieved/total*100:.0f}%)")
    
    # Temps moyen
    times = []
    for r in results:
        if 'execution_time' in r:
            time_str = r['execution_time']['total'].replace('s', '')
            try:
                times.append(float(time_str))
            except ValueError:
                pass
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"\nTemps moyen/ticket : {avg_time:.2f}s")
    
    # Résumé des escalades
    if index.ntotal > 0:
        print(f"\n🚨 ESCALADES STOCKÉES : {index.ntotal}")
        for doc, meta in zip(documents, metadata):
            print(f"   • {meta['ticket_id']}: {meta['reason'][:50]}...")
    
    print("\n" + "=" * 70)
    print("✅ Tests terminés !")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()