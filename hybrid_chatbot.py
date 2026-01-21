"""
Chatbot Hybride LSTM + LLM pour l'EIDIA-UEMF
Utilise un modèle LSTM pour la classification d'intentions
et un LLM (Llama 3, GPT, etc.) pour reformuler les réponses
"""

from chatbot import ChatbotModel
from llm_provider import LLMProvider, GeminiProvider
from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv
load_dotenv()

class HybridChatbot:
    """
    Chatbot hybride combinant classification LSTM et génération LLM
    
    Architecture:
    1. Question utilisateur → LSTM → Tag d'intention + Confiance
    2. Tag → JSON responses → Réponse brute
    3. Réponse brute → LLM → Réponse reformulée (naturelle)
    """
    
    # System Prompt optimisé pour l'EIDIA-UEMF
    SYSTEM_PROMPT = """Tu es l'assistant virtuel officiel de l'EIDIA (École Euro-Med d'Ingénierie Digitale et d'Intelligence Artificielle), qui fait partie de l'UEMF à Fès, Maroc.

🎯 TON RÔLE:
Tu aides les étudiants et candidats en reformulant les informations officielles de manière chaleureuse, claire et engageante, SANS JAMAIS inventer de nouvelles informations.

📋 RÈGLES STRICTES:
1. ✅ REFORMULE la réponse fournie pour la rendre plus humaine et conversationnelle
2. ✅ CONSERVE TOUTES les informations factuelles (dates, prix, noms, modules, salaires)
3. ✅ GARDE les émojis et la structure si elle aide à la lisibilité
4. ❌ N'INVENTE JAMAIS de nouvelles informations (modules, profs, débouchés, prix)
5. ❌ NE SUPPRIME AUCUNE information importante de la réponse originale
6. ✅ Adapte le ton selon le contexte (professionnel pour admission, plus léger pour vie étudiante)
7. ✅ Si la réponse contient des listes, garde-les claires et structurées

🎓 CONTEXTE EIDIA:
L'EIDIA forme des ingénieurs d'État en 5 ans (2 ans prépa + 3 ans ingénieur) dans 5 filières:
- Big Data & Analytique (Pr. Loubna Ourabah)
- Intelligence Artificielle (Pr. Asmae Abadi)
- Robotique & Cobotique (Pr. Bader El Kari)
- Cybersécurité & Computer Science (Pr. Taha)
- Full Stack Engineering & Multimédia (Pr. Mouhtadi Meryem)

💬 STYLE:
- Ton chaleureux mais professionnel
- Phrases courtes et claires
- Encourage l'étudiant dans son projet
- Termine par une ouverture si approprié

Reformule maintenant la réponse en respectant ces règles!"""

    def __init__(
        self, 
        lstm_model: Optional[ChatbotModel] = None,
        llm_provider: Optional[LLMProvider] = None,
        use_llm: bool = True
    ):
        """
        Initialise le chatbot hybride
        
        Args:
            lstm_model: Instance de ChatbotModel (si None, en crée une nouvelle)
            llm_provider: Provider LLM (Ollama, OpenAI, etc.)
            use_llm: Si False, utilise seulement le LSTM sans reformulation
        """
        # Initialiser le modèle LSTM
        self.lstm_model = lstm_model or ChatbotModel()
        print("📊 Chargement du modèle LSTM...")
        if not self.lstm_model.load_model():
            raise RuntimeError("Impossible de charger le modèle LSTM")
        self.lstm_model.load_responses()
        
        # Initialiser le provider LLM
        self.use_llm = use_llm
        if use_llm:
            self.llm_provider = llm_provider or self._auto_detect_provider()
        else:
            print("⚠️  Mode LSTM seul (pas de reformulation LLM)")
            self.llm_provider = None
    
    def _auto_detect_provider(self) -> LLMProvider:
        """Détecte automatiquement le provider Gemini"""
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            print("🔍 Gemini détecté, utilisation de Gemini Flash...")
            return GeminiProvider(api_key=gemini_key)
        else:
            raise ValueError(
                "❌ Clé API Gemini non trouvée!\n"
                "Définissez la variable d'environnement GEMINI_API_KEY"
            )
    
    def chat(
        self, 
        user_question: str, 
        reformulate: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Pipeline complet du chatbot hybride
        
        Args:
            user_question: Question de l'utilisateur
            reformulate: Si True, reformule avec le LLM
            temperature: Température du LLM (0.0 = déterministe, 1.0 = créatif)
            max_tokens: Nombre maximum de tokens générés
        
        Returns:
            Dict contenant:
                - question: Question originale
                - intent: Tag d'intention détecté
                - confidence: Confiance de la prédiction (0-1)
                - raw_response: Réponse brute du JSON
                - final_response: Réponse reformulée (ou brute si pas de LLM)
                - reformulated: True si reformulée par LLM
        """
        print(f"\n{'='*70}")
        print(f"💬 Question: {user_question}")
        print(f"{'='*70}")
        
        # Étape 1: Classification LSTM
        print("\n1️⃣ Classification LSTM...")
        lstm_result = self.lstm_model.chat_response(user_question)
        
        intent = lstm_result['intent']
        confidence = lstm_result['confidence']
        raw_response = lstm_result['response']
        
        print(f"   🎯 Intention: {intent}")
        print(f"   📊 Confiance: {confidence:.1%}")
        
        # Étape 2: Reformulation LLM (si activée et disponible)
        if reformulate and self.use_llm and self.llm_provider:
            print("\n2️⃣ Reformulation avec LLM...")
            try:
                final_response = self._reformulate_with_llm(
                    user_question=user_question,
                    intent=intent,
                    raw_response=raw_response,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                reformulated = True
                print("   ✅ Réponse reformulée avec succès")
            except Exception as e:
                print(f"   ⚠️  Erreur LLM: {e}")
                print("   ↩️  Retour à la réponse brute")
                final_response = raw_response
                reformulated = False
        else:
            final_response = raw_response
            reformulated = False
            print("\n2️⃣ Pas de reformulation (mode LSTM seul)")
        
        return {
            'question': user_question,
            'intent': intent,
            'confidence': confidence,
            'raw_response': raw_response,
            'final_response': final_response,
            'reformulated': reformulated
        }
    
    def _reformulate_with_llm(
        self,
        user_question: str,
        intent: str,
        raw_response: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        Reformule la réponse brute avec le LLM
        
        Args:
            user_question: Question originale de l'utilisateur
            intent: Tag d'intention détecté
            raw_response: Réponse brute du JSON
            temperature: Créativité du LLM
            max_tokens: Longueur maximale
        
        Returns:
            Réponse reformulée par le LLM
        """
        # Construire le message utilisateur pour le LLM
        user_message = f"""QUESTION DE L'ÉTUDIANT:
"{user_question}"

INTENTION DÉTECTÉE: {intent}

RÉPONSE BRUTE À REFORMULER:
{raw_response}

Reformule cette réponse de manière chaleureuse et naturelle tout en conservant TOUTES les informations factuelles."""

        # Appeler le LLM
        reformulated = self.llm_provider.generate(
            system_prompt=self.SYSTEM_PROMPT,
            user_message=user_message,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return reformulated.strip()
    
    def interactive(self, reformulate: bool = True):
        """
        Mode interactif du chatbot hybride
        
        Args:
            reformulate: Si True, utilise la reformulation LLM
        """
        mode = "HYBRIDE (LSTM + LLM)" if reformulate else "LSTM SEUL"
        print(f"\n{'='*70}")
        print(f"🤖 CHATBOT EIDIA-UEMF - MODE {mode}")
        print(f"{'='*70}")
        print("Tapez 'quit' pour quitter")
        print("Tapez 'toggle' pour changer de mode")
        print()
        
        while True:
            try:
                user_input = input("Vous: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Au revoir ! N'hésitez pas à revenir pour toute question sur l'EIDIA.")
                    break
                
                if user_input.lower() == 'toggle':
                    reformulate = not reformulate
                    mode = "HYBRIDE (LSTM + LLM)" if reformulate else "LSTM SEUL"
                    print(f"\n🔄 Passage en mode {mode}\n")
                    continue
                
                # Obtenir la réponse
                result = self.chat(user_input, reformulate=reformulate)
                
                # Afficher la réponse
                print(f"\n🤖 Bot [{result['intent']} - {result['confidence']:.0%}]:")
                print(result['final_response'])
                
                # Afficher si reformulé
                if result['reformulated']:
                    print("\n💡 (Réponse reformulée par LLM)")
                
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Au revoir !")
                break
            except Exception as e:
                print(f"\n❌ Erreur: {e}\n")


# Fonction utilitaire pour créer rapidement un chatbot
def create_hybrid_chatbot(
    model: str = None,
    api_key: str = None,
    use_llm: bool = True
) -> HybridChatbot:
    """
    Crée un chatbot hybride avec Gemini
    
    Args:
        model: Nom du modèle Gemini (gemini-3-flash-preview, gemini-1.5-pro, etc.)
        api_key: Clé API Gemini
        use_llm: Si False, utilise seulement le LSTM
    
    Returns:
        Instance de HybridChatbot
    """
    if not use_llm:
        return HybridChatbot(use_llm=False)
    
    llm = GeminiProvider(
        model=model or "gemini-3-flash-preview",
        api_key=api_key
    )
    
    return HybridChatbot(llm_provider=llm)


if __name__ == "__main__":
    # Exemple d'utilisation
    print("🚀 Initialisation du chatbot hybride EIDIA-UEMF...")
    
    # Créer le chatbot avec Gemini (utilise GEMINI_API_KEY du .env)
    chatbot = create_hybrid_chatbot()
    
    # Mode interactif
    chatbot.interactive(reformulate=True)
