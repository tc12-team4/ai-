# agents/triage_agent.py
from agno.agent import Agent
from agno.models.mistral import MistralChat
import json


class TriageAgent:
    """
    Agent de triage Doxa qui valide et classe les tickets
    Détecte : cohérence, intention, exceptions (émotions + données sensibles)
    """
    
    def __init__(self):
        self.agent = Agent(
            name="Doxa Triage Agent",
            model=MistralChat(id="mistral-small-latest"),
            description="Agent de triage pour support client Doxa",
            instructions=[
                "Tu es un agent de triage pour Doxa, une plateforme SaaS de gestion de projets.",
                "",
                "📋 TA MISSION :",
                "Analyser chaque ticket client et déterminer :",
                "1. COHERENCE : La question a-t-elle du sens ? (oui/non)",
                "2. INTENTION : Que veut le client ? (1 phrase)",
                "3. TYPE_QUESTION : bug|question|feedback|account",
                "4. EXCEPTIONS : Détecter émotions négatives et données sensibles",
                "",
                "⚠️ DÉTECTION EXCEPTIONS :",
                "",
                "A. ÉMOTIONS NÉGATIVES (escalade) :",
                "   - Colère : 'furieux', 'énervé', 'inacceptable'",
                "   - Frustration : 'encore', 'toujours', '3ème fois'",
                "   - Menaces : 'annuler', 'avocat'",
                "   → Ajoute 'emotion_negative' dans exceptions",
                "",
                "B. DONNÉES SENSIBLES (escalade critique) :",
                "   - Mots de passe : 'mon mdp est', 'password:'",
                "   - Cartes bancaires : 16 chiffres",
                "   - Téléphones : +213, 05XX, 06XX",
                "   → Ajoute 'data_sensitive' dans exceptions",
                "",
                "📤 FORMAT RÉPONSE (JSON strict) :",
                "{",
                '  "coherent": true/false,',
                '  "intention": "...",',
                '  "type_question": "bug|question|feedback|account",',
                '  "reasoning": "...",',
                '  "exceptions": []',
                "}"
            ],
            markdown=False
        )
    
    def analyze(self, question: str) -> dict:
        prompt = f"""Analyse ce ticket client Doxa :

QUESTION CLIENT :
"{question}"

Réponds UNIQUEMENT avec le JSON demandé."""

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
                result = self._validate_result(result)
                return result
            else:
                return self._fallback_response(question, "JSON non trouvé")
        
        except Exception as e:
            print(f"❌ Erreur Triage : {e}")
            return self._fallback_response(question, str(e))
    
    def _validate_result(self, result: dict) -> dict:
        if 'coherent' not in result:
            result['coherent'] = True
        if 'intention' not in result:
            result['intention'] = "Demande nécessite analyse"
        if 'type_question' not in result:
            result['type_question'] = 'question'
        if 'exceptions' not in result or not isinstance(result['exceptions'], list):
            result['exceptions'] = []
        if 'reasoning' not in result:
            result['reasoning'] = "Analyse automatique"
        return result
    
    def _fallback_response(self, question: str, error_msg: str) -> dict:
        exceptions = []
        question_lower = question.lower()
        
        mots_colere = ['furieux', 'énervé', 'inacceptable', 'catastrophique']
        if any(mot in question_lower for mot in mots_colere):
            exceptions.append('emotion_negative')
        
        if any(pattern in question_lower for pattern in ['mot de passe', 'password', 'mdp est']):
            exceptions.append('data_sensitive')
        
        return {
            "coherent": True,
            "intention": "Question nécessite traitement manuel",
            "type_question": "question",
            "reasoning": f"Fallback: {error_msg}",
            "exceptions": exceptions
        }