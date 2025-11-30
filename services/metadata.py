"""Module contenant les services pour les métadonnées de films et la traduction."""

import os
import requests
from typing import Optional, Dict, Any
from deep_translator import GoogleTranslator
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


class MetadataService:
    """Service pour récupérer les métadonnées de films via l'API OMDb."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialise le service de métadonnées.
        
        Args:
            api_key: Clé API OMDb. Si None, récupère depuis les variables d'environnement.
        """
        self.api_key = api_key or os.getenv('API_KEY_FILM')
        if not self.api_key:
            st.warning("Clé API OMDb non trouvée. Les métadonnées de films ne seront pas disponibles.")
    
    @st.cache_data(ttl=3600)
    def get_movie_data(self, title: str) -> Optional[Dict[str, Any]]:
        """
        Récupère les informations d'un film depuis l'API OMDb.
        
        Args:
            title: Titre du film (peut contenir l'année entre parenthèses)
        
        Returns:
            Dictionnaire contenant les métadonnées du film ou None en cas d'erreur.
            Les clés incluent: title, year, genre, director, actors, plot, rating, votes, poster
        """
        if not self.api_key:
            return None
        
        year = None
        clean_title = title
        
        # Extraction de l'année si présente dans le titre
        if title[-1] == ")" and "(" in title:
            try:
                year = title.split("(")[-1][:-1]
                clean_title = title.rsplit("(", 1)[0].strip()
            except:
                pass
        
        # Construction de l'URL
        url = f"http://www.omdbapi.com/?t={clean_title}&apikey={self.api_key}"
        if year:
            url += f"&y={year}"
        
        try:
            response = requests.get(url, timeout=10).json()
            if response.get("Response") == "True":
                return {
                    "title": response["Title"],
                    "year": response["Year"],
                    "genre": response.get("Genre", "N/A"),
                    "director": response.get("Director", "N/A"),
                    "actors": response.get("Actors", "N/A"),
                    "plot": response.get("Plot", "N/A"),
                    "rating": response.get("imdbRating", "N/A"),
                    "votes": response.get("imdbVotes", "0"),
                    "poster": response.get("Poster", None),
                }
        except requests.RequestException:
            return None
        
        return None


class TranslationService:
    """Service pour la traduction automatique avec cache."""
    
    # Configuration des langues disponibles
    LANGUAGES = {
        "fr": "🇫🇷 Français",
        "en": "🇬🇧 English",
        "es": "🇪🇸 Español",
        "de": "🇩🇪 Deutsch",
        "it": "🇮🇹 Italiano",
        "pt": "🇵🇹 Português",
        "ja": "🇯🇵 日本語",
        "zh-CN": "🇨🇳 中文",
        "ar": "🇸🇦 العربية",
        "ru": "🇷🇺 Русский"
    }
    
    def __init__(self, default_language: str = 'fr'):
        """
        Initialise le service de traduction.
        
        Args:
            default_language: Langue par défaut (code ISO)
        """
        self.default_language = default_language
        self._initialize_cache()
    
    def _initialize_cache(self):
        """Initialise le cache de traductions dans session_state si nécessaire."""
        if 'translations_cache' not in st.session_state:
            st.session_state.translations_cache = {}
    
    def translate(self, text: str, target_language: str) -> str:
        """
        Traduit un texte vers la langue cible avec mise en cache.
        
        Args:
            text: Texte à traduire (en français)
            target_language: Code de la langue cible
        
        Returns:
            Texte traduit ou texte original en cas d'erreur
        """
        if target_language == 'fr' or target_language == self.default_language:
            return text
        
        # Vérification du cache
        cache_key = f"{target_language}_{text}"
        if cache_key in st.session_state.translations_cache:
            return st.session_state.translations_cache[cache_key]
        
        # Traduction
        try:
            translated = GoogleTranslator(source='fr', target=target_language).translate(text)
            st.session_state.translations_cache[cache_key] = translated
            return translated
        except Exception:
            return text
    
    @classmethod
    def get_language_options(cls) -> Dict[str, str]:
        """
        Retourne le dictionnaire des langues disponibles.
        
        Returns:
            Dictionnaire {code_langue: nom_affiché}
        """
        return cls.LANGUAGES
    
    @classmethod
    def get_language_codes(cls) -> list:
        """
        Retourne la liste des codes de langues disponibles.
        
        Returns:
            Liste des codes de langues
        """
        return list(cls.LANGUAGES.keys())

