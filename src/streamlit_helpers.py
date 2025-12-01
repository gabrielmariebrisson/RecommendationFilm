"""Module contenant les helpers et fonctions d'initialisation pour l'application Streamlit."""

import streamlit as st
from typing import Callable

import pandas as pd

from src.core.recommender import MovieRecommender
from src.core.model_registry import ModelVersionManager
from src.services.metadata import MetadataService, TranslationService
from src.config import MODEL_REGISTRY_PATH


@st.cache_resource
def get_recommender() -> MovieRecommender:
    """
    Initialise et retourne le MovieRecommender.
    
    Tente de charger depuis le Model Registry si disponible,
    sinon utilise les chemins par défaut.
    """
    # Essayer de charger depuis le registre.
    registry = ModelVersionManager(registry_path=MODEL_REGISTRY_PATH)
    latest_version = registry.get_latest_stable_version()
    
    if latest_version:
        # Charger depuis le registre.
        recommender = MovieRecommender(
            model_registry=registry,
            model_version=latest_version,
        )
    else:
        # Fallback vers les chemins par défaut.
        recommender = MovieRecommender()
    
    if not recommender.initialize():
        st.error("Failed to initialize recommender. Check logs for details.")
        return recommender
    
    return recommender


@st.cache_resource
def get_metadata_service() -> MetadataService:
    """Initialise et retourne le MetadataService."""
    return MetadataService()


def get_translation_service() -> TranslationService:
    """Initialise et retourne le TranslationService."""
    return TranslationService()


def create_translation_function(
    translation_service: TranslationService,
    lang: str
) -> Callable[[str], str]:
    """
    Crée et retourne une fonction de traduction pour la langue spécifiée.
    
    Args:
        translation_service: Service de traduction
        lang: Code de langue (ex: 'fr', 'en')
    
    Returns:
        Fonction de traduction qui prend un texte et retourne sa traduction
    """
    def translate(text: str) -> str:
        """Fonction de traduction avec cache."""
        return translation_service.translate(text, lang)
    
    return translate


def has_selected_genre(genres_str: str, selected_genres: list) -> bool:
    """
    Vérifie si un film contient au moins un des genres sélectionnés.
    
    Args:
        genres_str: Chaîne de caractères contenant les genres séparés par '|'
        selected_genres: Liste des genres sélectionnés
    
    Returns:
        True si le film contient au moins un des genres sélectionnés, False sinon
    """
    if pd.isna(genres_str) or not genres_str:
        return False
    return any(g in str(genres_str) for g in selected_genres)

