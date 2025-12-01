"""Configuration centralisée de l'application."""

import os
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

# --- Chemins des fichiers ---
# BASE_DIR pointe vers la racine du projet (parent de src/).
BASE_DIR = Path(__file__).parent.parent.resolve()
TEMPLATES_DIR = BASE_DIR / "templates"
ASSETS_DIR = TEMPLATES_DIR / "assets"
FILM_DIR = ASSETS_DIR / "film"
IMAGES_DIR = ASSETS_DIR / "images"

# Chemins des fichiers du modèle
MODEL_PATH = FILM_DIR / "best_model.keras"
SCALER_USER_PATH = FILM_DIR / "scalerUser.pkl"
SCALER_ITEM_PATH = FILM_DIR / "scalerItem.pkl"
SCALER_TARGET_PATH = FILM_DIR / "scalerTarget.pkl"
MOVIE_DICT_PATH = FILM_DIR / "movie_dict.pkl"
ITEM_VECS_PATH = FILM_DIR / "item_vecs_finder.pkl"
UNIQUE_GENRES_PATH = FILM_DIR / "unique_genres.pkl"

# Chemins des images
NO_POSTER_IMAGE_PATH = IMAGES_DIR / "no-poster.jpg"
ARCHITECTURE_IMAGE_PATH = FILM_DIR / "architecture_model.png"

# --- Configuration API ---
API_KEY_FILM = os.getenv('API_KEY_FILM')
OMDB_API_URL = "http://www.omdbapi.com/"
OMDB_API_TIMEOUT = 10

# --- Configuration des langues ---
LANGUAGES: Dict[str, str] = {
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

DEFAULT_LANGUAGE = "fr"

# --- Configuration de l'application ---
APP_TITLE = "Ciné-Reco"
APP_LAYOUT = "wide"
PORTFOLIO_URL = "https://gabriel.mariebrisson.fr"

# --- Configuration du cache Streamlit ---
CACHE_TTL_SECONDS = 3600  # 1 heure.

# --- Configuration du Model Registry ---
MODEL_REGISTRY_PATH = BASE_DIR / "models"  # Dossier pour les versions de modèles.

