"""Module contenant les services pour les métadonnées de films et la traduction."""

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List

import aiohttp
import requests
from deep_translator import GoogleTranslator
import streamlit as st

from src.config import (
    API_KEY_FILM,
    OMDB_API_URL,
    OMDB_API_TIMEOUT,
    LANGUAGES,
    DEFAULT_LANGUAGE,
    CACHE_TTL_SECONDS,
)


class MetadataService:
    """Service pour récupérer les métadonnées de films via l'API OMDb."""
    
    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialise le service de métadonnées.
        
        Args:
            api_key: Clé API OMDb. Si None, récupère depuis les variables d'environnement.
        """
        self.api_key: Optional[str] = api_key or API_KEY_FILM
        if not self.api_key:
            st.warning("Clé API OMDb non trouvée. Les métadonnées de films ne seront pas disponibles.")
    
    @st.cache_data(ttl=CACHE_TTL_SECONDS)
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
        
        year: Optional[str] = None
        clean_title: str = title
        
        # Extraction de l'année si présente dans le titre
        if title and title[-1] == ")" and "(" in title:
            try:
                year = title.split("(")[-1][:-1]
                clean_title = title.rsplit("(", 1)[0].strip()
            except (IndexError, AttributeError) as e:
                # Si l'extraction échoue, on continue avec le titre original
                st.debug(f"Impossible d'extraire l'année du titre '{title}': {e}")
        
        # Construction de l'URL
        url: str = f"{OMDB_API_URL}?t={clean_title}&apikey={self.api_key}"
        if year:
            url += f"&y={year}"
        
        try:
            response: requests.Response = requests.get(url, timeout=OMDB_API_TIMEOUT)
            response.raise_for_status()
            data: Dict[str, Any] = response.json()
            
            if data.get("Response") == "True":
                return {
                    "title": data.get("Title", "N/A"),
                    "year": data.get("Year", "N/A"),
                    "genre": data.get("Genre", "N/A"),
                    "director": data.get("Director", "N/A"),
                    "actors": data.get("Actors", "N/A"),
                    "plot": data.get("Plot", "N/A"),
                    "rating": data.get("imdbRating", "N/A"),
                    "votes": data.get("imdbVotes", "0"),
                    "poster": data.get("Poster", None),
                }
        except requests.exceptions.Timeout:
            st.debug(f"Timeout lors de la requête OMDb pour '{title}'")
            return None
        except requests.exceptions.HTTPError as e:
            st.debug(f"Erreur HTTP lors de la requête OMDb pour '{title}': {e}")
            return None
        except requests.exceptions.RequestException as e:
            st.debug(f"Erreur de requête OMDb pour '{title}': {e}")
            return None
        except (KeyError, ValueError) as e:
            st.debug(f"Erreur lors du parsing de la réponse OMDb pour '{title}': {e}")
            return None
        
        return None
    
    async def get_movie_data_async(self, title: str, session: Optional[aiohttp.ClientSession] = None) -> Optional[Dict[str, Any]]:
        """
        Récupère asynchronement les informations d'un film depuis l'API OMDb.
        
        Args:
            title: Titre du film (peut contenir l'année entre parenthèses)
            session: Session aiohttp existante (optionnel, pour réutilisation)
        
        Returns:
            Dictionnaire contenant les métadonnées du film ou None en cas d'erreur.
            Les clés incluent: title, year, genre, director, actors, plot, rating, votes, poster
        """
        if not self.api_key:
            return None
        
        year: Optional[str] = None
        clean_title: str = title
        
        # Extraction de l'année si présente dans le titre.
        if title and title[-1] == ")" and "(" in title:
            try:
                year = title.split("(")[-1][:-1]
                clean_title = title.rsplit("(", 1)[0].strip()
            except (IndexError, AttributeError) as e:
                # Si l'extraction échoue, on continue avec le titre original.
                st.debug(f"Impossible d'extraire l'année du titre '{title}': {e}")
        
        # Construction de l'URL.
        url: str = f"{OMDB_API_URL}?t={clean_title}&apikey={self.api_key}"
        if year:
            url += f"&y={year}"
        
        # Utiliser la session fournie ou en créer une nouvelle.
        should_close_session = False
        if session is None:
            session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=OMDB_API_TIMEOUT))
            should_close_session = True
        
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                data: Dict[str, Any] = await response.json()
                
                if data.get("Response") == "True":
                    result = {
                        "title": data.get("Title", "N/A"),
                        "year": data.get("Year", "N/A"),
                        "genre": data.get("Genre", "N/A"),
                        "director": data.get("Director", "N/A"),
                        "actors": data.get("Actors", "N/A"),
                        "plot": data.get("Plot", "N/A"),
                        "rating": data.get("imdbRating", "N/A"),
                        "votes": data.get("imdbVotes", "0"),
                        "poster": data.get("Poster", None),
                    }
                    if should_close_session:
                        await session.close()
                    return result
        except asyncio.TimeoutError:
            st.debug(f"Timeout lors de la requête OMDb pour '{title}'")
            if should_close_session:
                await session.close()
            return None
        except aiohttp.ClientResponseError as e:
            st.debug(f"Erreur HTTP lors de la requête OMDb pour '{title}': {e}")
            if should_close_session:
                await session.close()
            return None
        except aiohttp.ClientError as e:
            st.debug(f"Erreur de requête OMDb pour '{title}': {e}")
            if should_close_session:
                await session.close()
            return None
        except (KeyError, ValueError) as e:
            st.debug(f"Erreur lors du parsing de la réponse OMDb pour '{title}': {e}")
            if should_close_session:
                await session.close()
            return None
        except Exception as e:
            st.debug(f"Erreur inattendue lors de la requête OMDb pour '{title}': {e}")
            if should_close_session:
                await session.close()
            return None
        
        if should_close_session:
            await session.close()
        return None
    
    async def get_movies_data_batch(self, titles: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Récupère asynchronement les métadonnées de plusieurs films en parallèle.
        
        Args:
            titles: Liste des titres de films à récupérer
        
        Returns:
            Dictionnaire {titre: métadonnées} avec None pour les films non trouvés
        """
        if not self.api_key or not titles:
            return {title: None for title in titles}
        
        # Filtrer les titres vides.
        valid_titles = [title for title in titles if title and title.strip()]
        if not valid_titles:
            return {title: None for title in titles}
        
        # Créer une session aiohttp partagée pour toutes les requêtes.
        timeout = aiohttp.ClientTimeout(total=OMDB_API_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Créer toutes les tâches asynchrones.
            tasks = [
                self.get_movie_data_async(title, session)
                for title in valid_titles
            ]
            
            # Exécuter toutes les requêtes en parallèle.
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Construire le dictionnaire de résultats.
        movies_data: Dict[str, Optional[Dict[str, Any]]] = {}
        for title, result in zip(valid_titles, results):
            if isinstance(result, Exception):
                st.debug(f"Erreur lors de la récupération de '{title}': {result}")
                movies_data[title] = None
            else:
                movies_data[title] = result
        
        # Ajouter None pour les titres invalides.
        for title in titles:
            if title not in movies_data:
                movies_data[title] = None
        
        return movies_data


class TranslationService:
    """Service pour la traduction automatique avec cache."""
    
    def __init__(self, default_language: str = DEFAULT_LANGUAGE) -> None:
        """
        Initialise le service de traduction.
        
        Args:
            default_language: Langue par défaut (code ISO)
        """
        self.default_language: str = default_language
        self._initialize_cache()
    
    def _initialize_cache(self) -> None:
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
        
        # Vérification du cache.
        cache_key: str = f"{target_language}_{text}"
        if cache_key in st.session_state.translations_cache:
            return st.session_state.translations_cache[cache_key]
        
        # Traduction.
        try:
            translator = GoogleTranslator(source='fr', target=target_language)
            translated: str = translator.translate(text)
            st.session_state.translations_cache[cache_key] = translated
            return translated
        except ValueError as e:
            st.debug(f"Langue de traduction invalide '{target_language}': {e}")
            return text
        except Exception as e:
            st.debug(f"Erreur lors de la traduction de '{text}' vers '{target_language}': {e}")
            return text
    
    @classmethod
    def get_language_options(cls) -> Dict[str, str]:
        """
        Retourne le dictionnaire des langues disponibles.
        
        Returns:
            Dictionnaire {code_langue: nom_affiché}
        """
        return LANGUAGES
    
    @classmethod
    def get_language_codes(cls) -> List[str]:
        """
        Retourne la liste des codes de langues disponibles.
        
        Returns:
            Liste des codes de langues
        """
        return list(LANGUAGES.keys())
