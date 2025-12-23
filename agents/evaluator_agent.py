# agents/evaluator_agent.py
"""
Agent Evaluator & Decider
Critère 4 (9%) : Confiance moyenne <0.6 → escalade
                  Détection émotions négatives, exceptions
"""

import json
from agno.agent import Agent
from agno.models.mistral import MistralChat

from dotenv import load_dotenv


load_dotenv()
class EvaluatorAgent:
    """
    Agent d'évaluation qui décide si la réponse est suffisamment fiable
    ou si une escalade vers un humain est nécessaire
    """
    
    def __init__(self):
        self.agent = Agent(
            name="Doxa Evaluator",
            model=MistralChat(id="mistral-small-latest", temperature=0.2),
            description="Agent d'évaluation de confiance et décision d'escalade",
            instructions=[
                "🎯 CONTEXTE : Tu es un évaluateur de qualité pour le support Doxa.",
                "",
                "📋 TA MISSION :",
                "Analyser la confiance des résultats de recherche et décider :",
                "- TRAITER : Confiance >= 0.6 (60%) → Peut générer une réponse",
                "- ESCALADER : Confiance < 0.6 (60%) → Doit passer à un humain",
                "",
                "📊 CRITÈRES D'ÉVALUATION :",
                "",
                "1. SCORE DE CONFIANCE (principal)",
                "   - Score >= 0.6 : Informations fiables",
                "   - Score < 0.6 : Informations insuffisantes ou peu pertinentes",
                "",
                "2. NOMBRE DE SOURCES",
                "   - 3+ sources différentes : Bon",
                "   - 1-2 sources : Acceptable si score > 0.7",
                "   - 0 source : Escalade automatique",
                "",
                "3. COHÉRENCE DES CHUNKS",
                "   - Chunks du même document : Cohérent",
                "   - Chunks de documents différents mais catégorie similaire : Acceptable",
                "   - Chunks sans rapport : Suspect → Escalade",
                "",
                "⚠️ CAS D'ESCALADE AUTOMATIQUE :",
                "- Confiance < 0.6",
                "- Aucune source trouvée",
                "- Question trop complexe nécessitant expertise humaine",
                "- Informations contradictoires entre sources",
                "",
                "📤 FORMAT DE RÉPONSE (JSON strict, sans markdown) :",
                "{",
                '  "decision": "traiter|escalader",',
                '  "confidence_finale": 0.85,',
                '  "raison": "Explication claire de la décision",',
                '  "recommandation": "Action recommandée",',
                '  "priorite_escalade": "basse|moyenne|haute"  // Seulement si escalade',
                "}",
                "",
                "🎓 EXEMPLES :",
                "",
                "Cas 1 : Score 0.75, 3 sources",
                '→ {"decision": "traiter", "confidence_finale": 0.75, "raison": "Score élevé, sources multiples"}',
                "",
                "Cas 2 : Score 0.45, 2 sources",
                '→ {"decision": "escalader", "confidence_finale": 0.45, "raison": "Score trop faible", "priorite_escalade": "moyenne"}',
                "",
                "Cas 3 : Score 0.82, 1 source unique",
                '→ {"decision": "traiter", "confidence_finale": 0.82, "raison": "Score excellent malgré source unique"}',
            ],
            markdown=False
        )
    
    def evaluate(
        self,
        retrieval_result: dict,
        query_data: dict = None
    ) -> dict:
        """
        Évalue si les résultats RAG sont suffisants pour générer une réponse
        
        Args:
            retrieval_result: Résultats du RAG
                {
                    "avg_score": 0.75,
                    "sources": ["doc1", "doc2"],
                    "num_chunks": 5,
                    "chunks": [...]
                }
            query_data: Données de la query (optionnel)
            
        Returns:
            dict avec decision, confidence_finale, raison, priorite_escalade
        """
        
        # Préparer les données d'entrée
        input_data = {
            "score_moyen": retrieval_result.get('avg_score', 0.0),
            "nombre_sources": len(retrieval_result.get('sources', [])),
            "sources": retrieval_result.get('sources', []),
            "nombre_chunks": retrieval_result.get('num_chunks', 0),
            "categorie": query_data.get('categorie', 'general') if query_data else 'general'
        }
        
        prompt = f"""Évalue cette récupération de documents :

DONNÉES DE RECHERCHE :
- Score moyen de confiance : {input_data['score_moyen']:.3f}
- Nombre de sources : {input_data['nombre_sources']}
- Sources : {', '.join(input_data['sources'])}
- Nombre de chunks récupérés : {input_data['nombre_chunks']}
- Catégorie de la question : {input_data['categorie']}

Fais ton évaluation et décide : traiter ou escalader ?
Réponds UNIQUEMENT avec le JSON demandé (pas de texte, pas de ```json)."""

        try:
            response = self.agent.run(prompt)
            content = response.content.strip()
            
            # Nettoyer markdown
            if content.startswith('```'):
                lines = content.split('\n')
                content = '\n'.join(lines[1:-1]) if len(lines) > 2 else content
            
            # Extraire JSON
            start = content.find('{')
            end = content.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = content[start:end]
                result = json.loads(json_str)
                
                # Validation
                result = self._validate_evaluation(result, input_data)
                return result
            else:
                print(f"⚠️ Pas de JSON dans l'évaluation")
                return self._fallback_evaluation(input_data)
        
        except Exception as e:
            print(f"❌ Erreur Evaluator : {e}")
            return self._fallback_evaluation(input_data)
    
    def _validate_evaluation(self, result: dict, input_data: dict) -> dict:
        """Valide et normalise l'évaluation"""
        
        # Décision valide
        if result.get('decision') not in ['traiter', 'escalader']:
            # Décision basée sur le score
            if input_data['score_moyen'] >= 0.6:
                result['decision'] = 'traiter'
            else:
                result['decision'] = 'escalader'
        
        # Confidence finale
        if 'confidence_finale' not in result:
            result['confidence_finale'] = input_data['score_moyen']
        
        # Raison
        if 'raison' not in result:
            if result['decision'] == 'traiter':
                result['raison'] = f"Score de confiance acceptable ({input_data['score_moyen']:.0%})"
            else:
                result['raison'] = f"Score de confiance insuffisant ({input_data['score_moyen']:.0%})"
        
        # Priorité escalade (si escalade)
        if result['decision'] == 'escalader':
            if 'priorite_escalade' not in result:
                if input_data['score_moyen'] < 0.3:
                    result['priorite_escalade'] = 'haute'
                elif input_data['score_moyen'] < 0.5:
                    result['priorite_escalade'] = 'moyenne'
                else:
                    result['priorite_escalade'] = 'basse'
        
        # Recommandation
        if 'recommandation' not in result:
            if result['decision'] == 'traiter':
                result['recommandation'] = "Générer une réponse automatique"
            else:
                result['recommandation'] = "Transférer vers un agent humain"
        
        return result
    
    def _fallback_evaluation(self, input_data: dict) -> dict:
        """Évaluation de secours basée sur des règles simples"""
        
        score = input_data['score_moyen']
        num_sources = input_data['nombre_sources']
        
        # Règle simple : score >= 0.6 → traiter
        if score >= 0.6 and num_sources > 0:
            return {
                "decision": "traiter",
                "confidence_finale": score,
                "raison": f"Score acceptable ({score:.0%}) avec {num_sources} source(s)",
                "recommandation": "Générer une réponse automatique"
            }
        else:
            priorite = 'haute' if score < 0.3 else 'moyenne' if score < 0.5 else 'basse'
            
            return {
                "decision": "escalader",
                "confidence_finale": score,
                "raison": f"Score insuffisant ({score:.0%}) ou manque de sources",
                "recommandation": "Transférer vers un agent humain",
                "priorite_escalade": priorite
            }


if __name__ == "__main__":
    # Test de l'agent
    evaluator = EvaluatorAgent()
    
    # Test 1 : Score élevé
    test1 = {
        "avg_score": 0.78,
        "sources": ["FAQ_Doxa", "Guide_Utilisateur"],
        "num_chunks": 5
    }
    
    result1 = evaluator.evaluate(test1)
    print("Test 1 (score élevé) :")
    print(json.dumps(result1, indent=2, ensure_ascii=False))
    
    # Test 2 : Score faible
    test2 = {
        "avg_score": 0.42,
        "sources": ["FAQ_Doxa"],
        "num_chunks": 3
    }
    
    result2 = evaluator.evaluate(test2)
    print("\nTest 2 (score faible) :")
    print(json.dumps(result2, indent=2, ensure_ascii=False))