"""
Module pour Google Gemini API
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import os


class LLMProvider(ABC):
    """Interface abstraite pour les providers LLM"""
    
    @abstractmethod
    def generate(self, system_prompt: str, user_message: str, **kwargs) -> str:
        """Génère une réponse à partir du prompt"""
        pass


class GeminiProvider(LLMProvider):
    """Provider pour Google Gemini API"""
    
    def __init__(self, model: str = "gemini-3-flash-preview", api_key: str = None):
        """
        Args:
            model: Nom du modèle Gemini (gemini-3-flash-preview, gemini-1.5-pro, etc.)
            api_key: Clé API Google
        """
        try:
            import google.generativeai as genai
            self.genai = genai
            
            if not api_key:
                import os
                api_key = os.getenv("GEMINI_API_KEY")
            
            if not api_key:
                raise ValueError("Clé API Gemini requise")
            
            self.genai.configure(api_key=api_key)
            self.model_name = model
            self.model = self.genai.GenerativeModel(model)
            print(f"✓ Gemini provider initialisé avec modèle: {model}")
        except ImportError:
            raise ImportError(
                "La bibliothèque 'google-generativeai' n'est pas installée.\n"
                "Installez-la avec: pip install google-generativeai"
            )
    
    def generate(self, system_prompt: str, user_message: str, **kwargs) -> str:
        """
        Génère une réponse avec Gemini
        
        Args:
            system_prompt: Instructions système pour le LLM
            user_message: Message de l'utilisateur
            **kwargs: Paramètres additionnels (temperature, max_tokens, etc.)
        
        Returns:
            Réponse générée par le LLM
        """
        try:
            # Gemini combine le system prompt et le message
            full_prompt = f"{system_prompt}\n\n{user_message}"
            
            generation_config = {
                "temperature": kwargs.get("temperature", 0.7),
                "max_output_tokens": kwargs.get("max_tokens", 4000),
            }
            
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            return response.text
        except Exception as e:
            raise RuntimeError(f"Erreur lors de l'appel à Gemini: {e}")
