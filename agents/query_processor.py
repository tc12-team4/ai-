# agents/query_processor.py
import json
from agno.agent import Agent
from agno.models.mistral import MistralChat


class SmartQueryProcessor:
    """
    Agent UNIFIÉ : Analyse + Classification Doxa
    """
    
    def __init__(self):
        self.agent = Agent(
            name="Doxa Smart Query Processor",
            model=MistralChat(id="mistral-small-latest"),
            description="Processeur intelligent de requêtes support Doxa",
            instructions=[
                "🎯 CONTEXTE : Doxa = plateforme SaaS gestion projets (Kanban, Agile, Waterfall).",
                "",
                "📋 TA MISSION :",
                "Produire un JSON avec :",
                "- Résumé (<100 mots)",
                "- 5-10 mots-clés",
                "- Reformulation claire",
                "- Catégorie précise",
                "- Documents KB suggérés",
                "- Score confiance",
                "",
                "🗂️ CATÉGORIES :",
                "1. technique (bugs, erreurs)",
                "2. authentification (login, 2FA)",
                "3. projets_taches (création, gestion)",
                "4. collaboration (commentaires, @mentions)",
                "5. integrations (Slack, GitHub, API)",
                "6. facturation (plans, paiement)",
                "7. securite (2FA, conformité)",
                "8. onboarding (démarrage)",
                "9. general (FAQ)",
                "",
                "📚 DOCUMENTS KB :",
                "- troubleshooting (bugs, erreurs)",
                "- guide_securite (2FA, permissions)",
                "- tarification (plans, prix)",
                "- guide_utilisateur (projets, tâches)",
                "- guide_onboarding (démarrage)",
                "- faq (questions générales)",
                "- conditions_generales (CGU, lois)",
                "",
                "🎯 MAPPING CATÉGORIE → DOCUMENTS :",
                "technique → troubleshooting + faq",
                "authentification → troubleshooting + guide_securite",
                "projets_taches → guide_utilisateur + troubleshooting",
                "collaboration → guide_utilisateur + faq",
                "integrations → guide_utilisateur + troubleshooting",
                "facturation → tarification + conditions_generales",
                "securite → guide_securite + conditions_generales",
                "onboarding → guide_onboarding + guide_utilisateur",
                "general → faq + guide_utilisateur",
                "",
                "📤 FORMAT RÉPONSE (JSON strict) :",
                "{",
                '  "resume": "...",',
                '  "mots_cles": ["mot1", "mot2", ...],',
                '  "reformulation": "...",',
                '  "categorie": "...",',
                '  "documents": ["doc1", "doc2"],',
                '  "confidence": 0.85,',
                '  "reasoning": "..."',
                "}"
            ],
            markdown=False
        )
    
    def process(self, question: str, triage_data: dict) -> dict:
        prompt = f"""Analyse cette requête support Doxa :

QUESTION ORIGINALE :
"{question}"

CONTEXTE DU TRIAGE :
- Intention : {triage_data.get('intention', 'N/A')}
- Type : {triage_data.get('type_question', 'question')}

Réponds UNIQUEMENT avec le JSON demandé."""

        try:
            response = self.agent.run(prompt)
            content = response.content.strip()
            
            if content.startswith('```'):
                lines = content.split('\n')
                content = '\n'.join(lines[1:-1]) if len(lines) > 2 else content
            
            start = content.find('{')
            end = content.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = content[start:end]
                result = json.loads(json_str)
                result['question_originale'] = question
                result = self._validate_response(result)
                return result
            else:
                return self._fallback_response(question, triage_data)
        
        except Exception as e:
            print(f"❌ Erreur Query Processor : {e}")
            return self._fallback_response(question, triage_data)
    
    def _validate_response(self, result: dict) -> dict:
        categories_valides = [
            'technique', 'authentification', 'projets_taches', 
            'collaboration', 'integrations', 'facturation', 
            'securite', 'onboarding', 'general'
        ]
        
        docs_valides = [
            'troubleshooting', 'guide_securite', 'tarification',
            'guide_utilisateur', 'guide_onboarding', 'faq',
            'conditions_generales'
        ]
        
        if result.get('categorie') not in categories_valides:
            result['categorie'] = 'general'
        
        if not result.get('documents'):
            result['documents'] = self._get_default_docs(result['categorie'])
        else:
            result['documents'] = [
                doc for doc in result['documents'] if doc in docs_valides
            ][:3]
            if not result['documents']:
                result['documents'] = ['faq']
        
        if not result.get('mots_cles'):
            result['mots_cles'] = ['support', 'question', 'doxa']
        else:
            mots_cles = result['mots_cles']
            if len(mots_cles) < 5:
                mots_cles.extend(['support', 'question', 'aide'])
            result['mots_cles'] = list(set(mots_cles[:10]))
        
        confidence = result.get('confidence', 0.5)
        result['confidence'] = max(0.0, min(1.0, float(confidence)))
        
        return result
    
    def _get_default_docs(self, categorie: str) -> list:
        mapping = {
            'technique': ['troubleshooting', 'faq'],
            'authentification': ['troubleshooting', 'guide_securite'],
            'projets_taches': ['guide_utilisateur', 'troubleshooting'],
            'collaboration': ['guide_utilisateur', 'faq'],
            'integrations': ['guide_utilisateur', 'troubleshooting'],
            'facturation': ['tarification', 'conditions_generales'],
            'securite': ['guide_securite', 'conditions_generales'],
            'onboarding': ['guide_onboarding', 'guide_utilisateur'],
            'general': ['faq', 'guide_utilisateur']
        }
        return mapping.get(categorie, ['faq'])
    
    def _fallback_response(self, question: str, triage_data: dict) -> dict:
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['login', 'connexion', 'mot de passe']):
            categorie = 'authentification'
        elif any(word in question_lower for word in ['erreur', 'bug', '500', '404']):
            categorie = 'technique'
        elif any(word in question_lower for word in ['plan', 'facture', 'paiement']):
            categorie = 'facturation'
        elif any(word in question_lower for word in ['projet', 'tâche', 'kanban']):
            categorie = 'projets_taches'
        else:
            categorie = 'general'
        
        return {
            "question_originale": question,
            "resume": question[:100] + "..." if len(question) > 100 else question,
            "mots_cles": ['support', 'question', 'doxa', 'aide'],
            "reformulation": question,
            "categorie": categorie,
            "documents": self._get_default_docs(categorie),
            "confidence": 0.5,
            "reasoning": "Fallback automatique"
        }