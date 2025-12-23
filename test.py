# main.py
"""
Script principal pour tester tous les agents du système Doxa Support AI
"""

import sys
from pathlib import Path
from agents.document_processor import DocumentProcessor
from agents.orchestrator import OrchestratorAgent


def test_document_processor():
    """Teste le Document Processor (chunking + FAISS)"""
    print("=" * 70)
    print("🧪 TEST 1 : Document Processor")
    print("=" * 70)
    
    # Créer un dossier test avec un PDF d'exemple
    test_pdf_dir = Path("./documents")
    test_pdf_dir.mkdir(exist_ok=True)
    
    # Créer un fichier texte factice pour le test (si pas de PDF disponible)
    if not list(test_pdf_dir.glob("*.pdf")):
        print("   ⚠️  Création d'un PDF factice pour le test...")
        from fpdf import FPDF
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Ajouter du contenu de test
        pdf.cell(200, 10, txt="FAQ Doxa - Support Technique", ln=1, align='C')
        pdf.ln(10)
        
        questions = [
            "Q: Comment activer 2FA ?",
            "R: Allez dans Profil > Sécurité > Activer 2FA",
            "",
            "Q: Comment créer un projet ?",
            "R: Cliquez sur Nouveau projet dans le tableau de bord",
            "",
            "Q: Problème de connexion ?",
            "R: Vérifiez vos identifiants ou réinitialisez votre mot de passe"
        ]
        
        for line in questions:
            pdf.cell(0, 10, txt=line, ln=1)
        
        pdf_path = test_pdf_dir / "test_faq.pdf"
        pdf.output(str(pdf_path))
        print(f"   ✅ PDF créé : {pdf_path}")
    
    # Initialiser le Document Processor
    doc_processor = DocumentProcessor(
        pdf_dir="./test_pdfs",
        md_dir="./test_md",
        index_dir="./test_index",
        chunk_size=300,
        overlap_sentences=1
    )
    
    # Option 1 : Charger un index existant
    print("\n   🔍 Tentative de chargement d'index existant...")
    if doc_processor.load_index():
        print(f"   ✅ Index chargé : {doc_processor.index.ntotal} vecteurs")
    else:
        # Option 2 : Créer un nouveau index
        print("   📝 Création d'un nouvel index...")
        doc_processor.process()
    
    return doc_processor


def test_triage_agent(doc_processor):
    """Teste l'agent de triage"""
    print("\n" + "=" * 70)
    print("🧪 TEST 2 : Triage Agent")
    print("=" * 70)
    
    from agents.triage_agent import TriageAgent
    
    triage = TriageAgent()
    
    test_cases = [
        {
            "id": "T1",
            "question": "Comment je peux activer la 2FA sur mon compte ?",
            "expected": "coherent"
        },
        {
            "id": "T2",
            "question": "Je suis furieux ! Mon projet a disparu !",
            "expected": "emotion_negative"
        },
        {
            "id": "T3",
            "question": "Mon mot de passe est: SuperSecret123",
            "expected": "data_sensitive"
        },
        {
            "id": "T4",
            "question": "blabla blabla pas de sens",
            "expected": "incoherent"
        }
    ]
    
    for test in test_cases:
        print(f"\n   🔹 Test {test['id']} : {test['question'][:50]}...")
        result = triage.analyze(test['question'])
        
        print(f"     ✅ Cohérent : {'OUI' if result['coherent'] else 'NON'}")
        print(f"     📋 Type : {result['type_question']}")
        print(f"     🎯 Intention : {result['intention'][:50]}...")
        
        if result['exceptions']:
            print(f"     ⚠️  Exceptions : {', '.join(result['exceptions'])}")
    
    return triage


def test_query_processor():
    """Teste le Query Processor"""
    print("\n" + "=" * 70)
    print("🧪 TEST 3 : Query Processor")
    print("=" * 70)
    
    from agents.query_processor import SmartQueryProcessor
    from agents.triage_agent import TriageAgent
    
    triage = TriageAgent()
    query_processor = SmartQueryProcessor()
    
    test_questions = [
        "Je ne peux pas me connecter à mon compte",
        "Comment créer un nouveau projet Kanban ?",
        "Combien coûte le plan premium ?",
        "Mon application plante quand j'ajoute une tâche"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n   🔹 Question {i} : {question}")
        
        # D'abord le triage
        triage_result = triage.analyze(question)
        
        # Puis le query processing
        query_data = query_processor.process(question, triage_result)
        
        print(f"     📋 Catégorie : {query_data['categorie']}")
        print(f"     🎯 Résumé : {query_data['resume'][:60]}...")
        print(f"     🔑 Mots-clés : {', '.join(query_data['mots_cles'][:5])}")
        print(f"     📚 Documents suggérés : {', '.join(query_data['documents'])}")
        print(f"     📊 Confiance : {query_data['confidence']:.0%}")
    
    return query_processor


def test_evaluator_agent():
    """Teste l'Evaluator Agent"""
    print("\n" + "=" * 70)
    print("🧪 TEST 4 : Evaluator Agent")
    print("=" * 70)
    
    from agents.evaluator_agent import EvaluatorAgent
    
    evaluator = EvaluatorAgent()
    
    test_cases = [
        {
            "name": "Score élevé",
            "data": {
                "avg_score": 0.85,
                "sources": ["FAQ_Doxa", "Guide_Utilisateur", "Troubleshooting"],
                "num_chunks": 7
            }
        },
        {
            "name": "Score moyen",
            "data": {
                "avg_score": 0.65,
                "sources": ["FAQ_Doxa"],
                "num_chunks": 3
            }
        },
        {
            "name": "Score faible",
            "data": {
                "avg_score": 0.42,
                "sources": ["FAQ_Doxa"],
                "num_chunks": 2
            }
        },
        {
            "name": "Aucune source",
            "data": {
                "avg_score": 0.0,
                "sources": [],
                "num_chunks": 0
            }
        }
    ]
    
    for test in test_cases:
        print(f"\n   🔹 {test['name']} :")
        result = evaluator.evaluate(test['data'])
        
        print(f"     ⚖️  Décision : {result['decision'].upper()}")
        print(f"     📊 Confiance : {result['confidence_finale']:.0%}")
        print(f"     📝 Raison : {result['raison']}")
        
        if result['decision'] == 'escalader':
            print(f"     🚨 Priorité : {result.get('priorite_escalade', 'N/A').upper()}")
    
    return evaluator


def test_response_composer():
    """Teste le Response Composer"""
    print("\n" + "=" * 70)
    print("🧪 TEST 5 : Response Composer")
    print("=" * 70)
    
    from agents.response_composer import ResponseComposer
    
    composer = ResponseComposer()
    
    test_cases = [
        {
            "langue": "français",
            "question": "Comment activer 2FA ?",
            "context": """
            Pour activer 2FA sur votre compte Doxa :
            1. Connectez-vous à votre compte
            2. Allez dans Paramètres > Sécurité
            3. Cliquez sur "Activer 2FA"
            4. Scannez le QR code avec Google Authenticator
            5. Sauvegardez les codes de secours
            
            Important : Les codes de secours sont essentiels si vous perdez votre téléphone.
            """
        },
        {
            "langue": "english",
            "question": "How to create a new project?",
            "context": """
            Creating a new project in Doxa:
            - Click the "+ New Project" button in dashboard
            - Choose project type: Kanban, Agile, or Waterfall
            - Add project name and description
            - Set start and end dates
            - Invite team members
            
            You can later customize columns, workflows, and notifications.
            """
        }
    ]
    
    for test in test_cases:
        print(f"\n   🔹 Test {test['langue']} :")
        print(f"     ❓ Question : {test['question']}")
        
        result = composer.compose(
            question=test['question'],
            context=test['context']
        )
        
        print(f"     🌍 Langue : {result['langue']}")
        print(f"     📊 Qualité : {result['quality_score']:.0%}")
        print(f"     🔢 Mots : {result['word_count']}")
        
        if result['quality_issues']:
            print(f"     ⚠️  Problèmes : {', '.join(result['quality_issues'])}")
        
        print(f"\n     📝 Réponse :\n{'─' * 40}")
        print(result['response'])
        print(f"{'─' * 40}")
    
    return composer


def test_retrieval_agent(doc_processor):
    """Teste le Retrieval Agent"""
    print("\n" + "=" * 70)
    print("🧪 TEST 6 : Retrieval Agent")
    print("=" * 70)
    
    from agents.retrieval_agent import RetrievalAgent
    
    retrieval = RetrievalAgent(doc_processor)
    
    test_queries = [
        {
            "name": "Recherche standard",
            "query_data": {
                "reformulation": "Comment activer sécurité 2FA",
                "categorie": "securite",
                "documents": ["guide_securite", "faq"]
            }
        },
        {
            "name": "Recherche avec mots-clés",
            "query_data": {
                "reformulation": "problème connexion compte",
                "mots_cles": ["login", "erreur", "connexion"],
                "categorie": "authentification",
                "documents": ["troubleshooting", "guide_securite"]
            }
        }
    ]
    
    for test in test_queries:
        print(f"\n   🔹 {test['name']} :")
        print(f"     🔍 Recherche : {test['query_data']['reformulation']}")
        
        try:
            result = retrieval.retrieve(
                query_data=test['query_data'],
                top_k=3,
                use_augmentation=True
            )
            
            print(f"     📊 Score moyen : {result['avg_score']:.3f}")
            print(f"     📚 Sources : {', '.join(result['sources'])}")
            print(f"     ✂️  Chunks : {result['num_chunks']}")
            
            if result['chunks']:
                print(f"\n     📄 Premier chunk :")
                print(f"       Source : {result['chunks'][0]['source']}")
                print(f"       Score : {result['chunks'][0]['score']:.3f}")
                print(f"       Texte : {result['chunks'][0]['text'][:100]}...")
        
        except Exception as e:
            print(f"     ❌ Erreur : {e}")
    
    return retrieval


def test_orchestrator_complet(doc_processor):
    """Teste l'orchestrateur complet"""
    print("\n" + "=" * 70)
    print("🧪 TEST 7 : Orchestrator Complet")
    print("=" * 70)
    
    print("\n   🚀 Initialisation de l'orchestrateur...")
    
    try:
        orchestrator = OrchestratorAgent(doc_processor)
        
        # Tests de tickets
        test_tickets = [
            {
                "id": "TICKET-001",
                "question": "Bonjour, je souhaite activer la double authentification sur mon compte Doxa. Pouvez-vous m'aider ?",
                "description": "Question normale - devrait être traitée"
            },
            {
                "id": "TICKET-002", 
                "question": "JE SUIS FURIEUX ! Mon projet a disparu après la mise à jour ! C'est INACCEPTABLE !",
                "description": "Émotion négative - devrait escalader"
            },
            {
                "id": "TICKET-003",
                "question": "Mon mot de passe est secret123, pouvez-vous vérifier ?",
                "description": "Donnée sensible - devrait escalader immédiatement"
            },
            {
                "id": "TICKET-004",
                "question": "Comment créer un tableau Kanban avec des colonnes personnalisées ?",
                "description": "Question technique - devrait être traitée"
            }
        ]
        
        for ticket in test_tickets[:2]:  # Tester seulement 2 tickets pour la démo
            print(f"\n{'=' * 70}")
            print(f"🎫 Traitement du ticket : {ticket['id']}")
            print(f"📝 Description : {ticket['description']}")
            print(f"{'=' * 70}")
            
            result = orchestrator.process_ticket(
                ticket_id=ticket['id'],
                question=ticket['question']
            )
            
            print(f"\n   📊 RÉSULTAT :")
            print(f"     ✅ Status : {result['status'].upper()}")
            
            if result['status'] == 'completed':
                print(f"     📝 Réponse générée ({result['response_data']['word_count']} mots)")
                print(f"     🌍 Langue : {result['response_data']['langue']}")
                print(f"     📊 Qualité : {result['response_data']['quality_score']:.0%}")
            elif result['status'] == 'escalated':
                print(f"     🚨 Raison : {result['reason']}")
                print(f"     ⚠️  Priorité : {result['priority']}")
            elif result['status'] == 'rejected':
                print(f"     ❌ Message : {result['message']}")
            
            print(f"     ⏱️  Temps total : {result['execution_time']['total']}")
    
    except Exception as e:
        print(f"   ❌ Erreur lors de l'initialisation : {e}")
        import traceback
        traceback.print_exc()
    
    return None


def menu_principal():
    """Affiche un menu interactif pour tester les agents"""
    doc_processor = None
    
    while True:
        print("\n" + "=" * 70)
        print("🤖 SYSTÈME DOXA SUPPORT AI - MENU DE TEST")
        print("=" * 70)
        print("1. 🧪 Tester Document Processor (FAISS)")
        print("2. 🎯 Tester Triage Agent")
        print("3. 🔍 Tester Query Processor")
        print("4. ⚖️  Tester Evaluator Agent")
        print("5. 📝 Tester Response Composer")
        print("6. 🔎 Tester Retrieval Agent")
        print("7. 🚀 Tester Orchestrator Complet")
        print("8. 📊 Tester Tous les Agents")
        print("0. ❌ Quitter")
        print("-" * 70)
        
        choix = input("👉 Sélectionnez une option (0-8) : ").strip()
        
        try:
            if choix == "1":
                doc_processor = test_document_processor()
                input("\n🎯 Appuyez sur Entrée pour continuer...")
            
            elif choix == "2":
                if doc_processor is None:
                    print("⚠️  Document Processor non chargé. Exécutez d'abord le test 1.")
                    doc_processor = test_document_processor()
                test_triage_agent(doc_processor)
                input("\n🎯 Appuyez sur Entrée pour continuer...")
            
            elif choix == "3":
                test_query_processor()
                input("\n🎯 Appuyez sur Entrée pour continuer...")
            
            elif choix == "4":
                test_evaluator_agent()
                input("\n🎯 Appuyez sur Entrée pour continuer...")
            
            elif choix == "5":
                test_response_composer()
                input("\n🎯 Appuyez sur Entrée pour continuer...")
            
            elif choix == "6":
                if doc_processor is None:
                    print("⚠️  Document Processor non chargé. Exécutez d'abord le test 1.")
                    doc_processor = test_document_processor()
                test_retrieval_agent(doc_processor)
                input("\n🎯 Appuyez sur Entrée pour continuer...")
            
            elif choix == "7":
                if doc_processor is None:
                    print("⚠️  Document Processor non chargé. Exécutez d'abord le test 1.")
                    doc_processor = test_document_processor()
                test_orchestrator_complet(doc_processor)
                input("\n🎯 Appuyez sur Entrée pour continuer...")
            
            elif choix == "8":
                print("\n🚀 Exécution de tous les tests...")
                doc_processor = test_document_processor()
                test_triage_agent(doc_processor)
                test_query_processor()
                test_evaluator_agent()
                test_response_composer()
                test_retrieval_agent(doc_processor)
                test_orchestrator_complet(doc_processor)
                print("\n✅ Tous les tests sont terminés !")
                input("\n🎯 Appuyez sur Entrée pour continuer...")
            
            elif choix == "0":
                print("\n👋 Au revoir !")
                sys.exit(0)
            
            else:
                print("❌ Option invalide. Veuillez choisir entre 0 et 8.")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interruption par l'utilisateur.")
            continue
        except Exception as e:
            print(f"\n❌ Erreur : {e}")
            import traceback
            traceback.print_exc()
            input("\n🎯 Appuyez sur Entrée pour continuer...")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🤖 SYSTÈME DOXA SUPPORT AI")
    print("📚 Agents : Document Processor, Triage, Query, Retrieval, Evaluator, Response")
    print("=" * 70)
    
    # Vérifier les dépendances
    try:
        import faiss
        import agno
        from sentence_transformers import SentenceTransformer
        print("✅ Toutes les dépendances sont installées")
    except ImportError as e:
        print(f"❌ Dépendance manquante : {e}")
        print("\n📦 Installation des dépendances requises :")
        print("pip install faiss-cpu sentence-transformers agno")
        sys.exit(1)
    
    # Installer fpdf si nécessaire pour créer le PDF de test
    try:
        import fpdf
    except ImportError:
        print("\n⚠️  Le module 'fpdf' n'est pas installé.")
        print("Il est nécessaire pour créer un PDF de test.")
        choix = input("Voulez-vous l'installer ? (o/n) : ").lower()
        if choix == 'o':
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf"])
            print("✅ fpdf installé avec succès")
        else:
            print("⚠️  Certains tests nécessiteront un PDF existant dans ./test_pdfs/")
    
    # Lancer le menu interactif
    menu_principal()