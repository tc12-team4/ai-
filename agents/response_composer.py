# agents/response_composer.py
"""
Agent Response Composer
Critère 6 (9%) : Template structuré + détection langue
Critère 7 (8%) : Qualité réponses (zéro hallucination, ton professionnel)
"""

import json
import re
from agno.agent import Agent
from agno.models.mistral import MistralChat
from dotenv import load_dotenv


load_dotenv()

class ResponseComposer:
    """
    Agent de composition de réponses structurées pour le support Doxa
    Génère des réponses dans la langue du client avec template strict
    """
    
    def __init__(self):
        self.agent = Agent(
            name="Doxa Response Composer",
            model=MistralChat(id="mistral-small-latest", temperature=0.3),
            description="Agent de génération de réponses support structurées",
            instructions=[
                "🎯 CONTEXTE : Tu es un agent de support Doxa qui génère des réponses clients.",
                "",
                "📋 TA MISSION :",
                "Générer une réponse structurée, professionnelle et empathique",
                "en utilisant UNIQUEMENT les informations du contexte fourni.",
                "",
                "🌍 DÉTECTION DE LANGUE :",
                "1. Détecte la langue de la question (français, anglais, arabe)",
                "2. Réponds dans LA MÊME LANGUE",
                "3. Si incertain → utilise le français",
                "",
                "📝 STRUCTURE OBLIGATOIRE (4 sections) :",
                "",
                "1. REMERCIEMENTS (1-2 phrases)",
                "   - Remercie le client pour sa demande",
                "   - Ton chaleureux et professionnel",
                "   Exemple FR : 'Merci d'avoir contacté le support Doxa.'",
                "   Exemple EN : 'Thank you for contacting Doxa support.'",
                "",
                "2. PROBLÈME (2-3 phrases)",
                "   - Reformule le problème du client",
                "   - Montre que tu as bien compris",
                "   - Empathie si nécessaire",
                "   Exemple : 'Vous rencontrez une difficulté pour...'",
                "",
                "3. SOLUTION (3-5 bullets)",
                "   - Étapes concrètes et actionnables",
                "   - Basées UNIQUEMENT sur le contexte fourni",
                "   - Ordre logique",
                "   - Utilisez des bullet points (•)",
                "   Exemple :",
                "   • Étape 1 : Allez dans Paramètres",
                "   • Étape 2 : Cliquez sur Sécurité",
                "",
                "4. ACTIONS (1-2 bullets)",
                "   - Ce que le client doit faire maintenant",
                "   - Contact support si besoin",
                "   Exemple :",
                "   • Essayez ces étapes",
                "   • Si le problème persiste : support@doxa.dz",
                "",
                "⚠️ RÈGLES CRITIQUES :",
                "- ZÉRO HALLUCINATION : N'invente RIEN, utilise UNIQUEMENT le contexte",
                "- Si info manquante dans contexte → Dis 'Information non disponible'",
                "- Ton professionnel mais chaleureux",
                "- Max 200 mots total",
                "- Pas de markdown (```), pas de titres de sections visibles",
                "",
                "❌ INTERDICTIONS :",
                "- Ne pas inventer des fonctionnalités",
                "- Ne pas donner des prix si pas dans contexte",
                "- Ne pas promettre des délais",
                "- Ne pas écrire les titres (Remerciements:, Problème:, etc.)",
                "",
                "📤 FORMAT DE SORTIE :",
                "Texte en prose naturelle avec les 4 sections intégrées",
                "Pas de JSON, pas de markdown, juste le texte de la réponse",
                "",
                "🎓 EXEMPLE COMPLET (FR) :",
                "",
                "Question : 'Comment activer 2FA ?'",
                "Contexte : [Extrait guide sécurité avec étapes 2FA]",
                "",
                "Réponse générée :",
                "Merci d'avoir contacté le support Doxa. Nous sommes là pour vous aider.",
                "",
                "Vous souhaitez activer l'authentification à deux facteurs (2FA) pour sécuriser votre compte. C'est une excellente pratique de sécurité.",
                "",
                "Voici comment procéder :",
                "• Accédez à votre Profil en haut à droite",
                "• Cliquez sur 'Paramètres' puis 'Sécurité'",
                "• Activez l'option '2FA' et scannez le QR code avec votre application d'authentification",
                "• Sauvegardez les codes de secours en lieu sûr",
                "",
                "Actions recommandées :",
                "• Suivez ces étapes pour activer 2FA",
                "• Si vous rencontrez un problème : contactez security@doxa.dz",
            ],
            markdown=False
        )
    
    def compose(
        self,
        question: str,
        context: str,
        query_data: dict = None,
        evaluation: dict = None
    ) -> dict:
        """
        Compose une réponse structurée
        
        Args:
            question: Question originale du client
            context: Contexte RAG (extraits de documents)
            query_data: Données de l'analyse de query (optionnel)
            evaluation: Résultat de l'évaluation (optionnel)
            
        Returns:
            dict avec response, langue, confidence
        """
        
        # Détecter la langue
        langue = self._detect_language(question)
        
        # Préparer le prompt
        prompt = f"""QUESTION DU CLIENT :
"{question}"

CONTEXTE DISPONIBLE (extraits documentation Doxa) :
{context}

CATÉGORIE : {query_data.get('categorie', 'general') if query_data else 'general'}

INSTRUCTIONS :
Génère une réponse structurée en {langue} selon le template à 4 sections.
Utilise UNIQUEMENT les informations du contexte ci-dessus.
N'invente RIEN.

Réponds directement avec le texte de la réponse (pas de JSON, pas de ```).
"""

        try:
            response = self.agent.run(prompt)
            response_text = response.content.strip()
            
            # Nettoyer markdown si présent
            response_text = re.sub(r'```.*?```', '', response_text, flags=re.DOTALL)
            response_text = response_text.strip()
            
            # Validation qualité
            quality_check = self._check_quality(response_text, context)
            
            return {
                "response": response_text,
                "langue": langue,
                "quality_score": quality_check['score'],
                "quality_issues": quality_check['issues'],
                "word_count": len(response_text.split()),
                "has_structure": self._has_structure(response_text)
            }
        
        except Exception as e:
            print(f"❌ Erreur ResponseComposer : {e}")
            return self._fallback_response(question, langue)
    
    def _detect_language(self, text: str) -> str:
        """
        Détecte la langue du texte
        
        Returns:
            'français', 'english', ou 'arabe'
        """
        
        # Patterns simples pour détecter la langue
        french_words = ['comment', 'je', 'ne', 'pas', 'mon', 'ma', 'mes', 'le', 'la', 'les', 'est', 'puis']
        english_words = ['how', 'can', 'my', 'the', 'is', 'are', 'not', 'what', 'why', 'where']
        
        text_lower = text.lower()
        
        french_count = sum(1 for word in french_words if word in text_lower)
        english_count = sum(1 for word in english_words if word in text_lower)
        
        # Détection arabe (caractères Unicode)
        arabic_chars = len([c for c in text if '\u0600' <= c <= '\u06FF'])
        
        if arabic_chars > 5:
            return 'arabe'
        elif french_count > english_count:
            return 'français'
        elif english_count > 0:
            return 'english'
        else:
            return 'français'  # Défaut
    
    def _check_quality(self, response: str, context: str) -> dict:
        """
        Vérifie la qualité de la réponse générée
        
        Returns:
            dict avec score (0-1) et liste des problèmes
        """
        
        issues = []
        score = 1.0
        
        # Check 1 : Longueur raisonnable
        word_count = len(response.split())
        if word_count < 50:
            issues.append("Réponse trop courte")
            score -= 0.2
        elif word_count > 300:
            issues.append("Réponse trop longue")
            score -= 0.1
        
        # Check 2 : Présence de sections
        if not self._has_structure(response):
            issues.append("Structure manquante")
            score -= 0.3
        
        # Check 3 : Mots interdits (hallucination courante)
        hallucination_words = ['peut-être', 'probablement', 'je pense', 'il semble']
        if any(word in response.lower() for word in hallucination_words):
            issues.append("Hallucination possible détectée")
            score -= 0.2
        
        # Check 4 : Référence au contexte
        # (Check si quelques mots du contexte sont dans la réponse)
        context_words = set(context.lower().split()[:50])
        response_words = set(response.lower().split())
        overlap = len(context_words & response_words)
        
        if overlap < 5:
            issues.append("Réponse semble détachée du contexte")
            score -= 0.2
        
        score = max(0.0, min(1.0, score))
        
        return {
            "score": score,
            "issues": issues
        }
    
    def _has_structure(self, response: str) -> bool:
        """Vérifie si la réponse a une structure (bullets)"""
        return '•' in response or '-' in response or response.count('\n') >= 3
    
    def _fallback_response(self, question: str, langue: str) -> dict:
        """Réponse de secours en cas d'erreur"""
        
        templates = {
            'français': (
                "Merci d'avoir contacté le support Doxa.\n\n"
                "Nous avons bien reçu votre demande concernant votre question. "
                "Notre équipe analyse votre situation.\n\n"
                "Actions recommandées :\n"
                "• Contactez support@doxa.dz pour assistance immédiate\n"
                "• Consultez notre documentation : docs.doxa.dz"
            ),
            'english': (
                "Thank you for contacting Doxa support.\n\n"
                "We have received your request. "
                "Our team is analyzing your situation.\n\n"
                "Recommended actions:\n"
                "• Contact support@doxa.dz for immediate assistance\n"
                "• Check our documentation: docs.doxa.dz"
            ),
            'arabe': (
                "شكرا لتواصلك مع دعم Doxa.\n\n"
                "تم استلام طلبك. فريقنا يحلل وضعك.\n\n"
                "الإجراءات الموصى بها:\n"
                "• اتصل بـ support@doxa.dz للمساعدة الفورية\n"
                "• راجع وثائقنا: docs.doxa.dz"
            )
        }
        
        return {
            "response": templates.get(langue, templates['français']),
            "langue": langue,
            "quality_score": 0.5,
            "quality_issues": ["Réponse de secours utilisée"],
            "word_count": 50,
            "has_structure": True
        }


if __name__ == "__main__":
    # Test de l'agent
    composer = ResponseComposer()
    
    # Test 1 : Question en français
    question = "Comment activer 2FA sur mon compte ?"
    context = """
    [Extrait 1] Source : guide_securite.pdf
    Pour activer 2FA :
    1. Allez dans Profil → Paramètres → Sécurité
    2. Activez l'option 2FA
    3. Scannez le QR code avec Google Authenticator
    4. Sauvegardez les codes de secours
    """
    
    result = composer.compose(question, context)
    
    print("Test 1 (français) :")
    print(f"Langue détectée : {result['langue']}")
    print(f"Qualité : {result['quality_score']:.0%}")
    print(f"Mots : {result['word_count']}")
    print(f"\nRéponse :\n{result['response']}")