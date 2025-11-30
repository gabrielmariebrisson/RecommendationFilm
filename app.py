"""Application Streamlit principale pour le système de recommandation de films."""

import asyncio
from typing import List, Set, Dict

import streamlit as st
import pandas as pd

from core.recommender import MovieRecommender
from services.metadata import MetadataService, TranslationService
from config import (
    NO_POSTER_IMAGE_PATH,
    ARCHITECTURE_IMAGE_PATH,
    APP_TITLE,
    APP_LAYOUT,
    PORTFOLIO_URL,
    DEFAULT_LANGUAGE,
)


# Initialisation des services
@st.cache_resource
def get_recommender() -> MovieRecommender:
    """Initialise et retourne le MovieRecommender."""
    recommender = MovieRecommender()
    recommender.initialize()
    return recommender


@st.cache_resource
def get_metadata_service() -> MetadataService:
    """Initialise et retourne le MetadataService."""
    return MetadataService()


def get_translation_service() -> TranslationService:
    """Initialise et retourne le TranslationService."""
    return TranslationService()


# Configuration de la langue
if 'language' not in st.session_state:
    st.session_state.language = DEFAULT_LANGUAGE

translation_service: TranslationService = get_translation_service()
lang_options: Dict[str, str] = translation_service.get_language_options()
lang_codes: List[str] = translation_service.get_language_codes()

# Sélecteur de langue
lang: str = st.sidebar.selectbox(
    "🌐 Language / Langue",
    options=lang_codes,
    format_func=lambda x: lang_options[x],
    index=lang_codes.index(st.session_state.language) if st.session_state.language in lang_codes else 0
)

st.session_state.language = lang


def _(text: str) -> str:
    """Fonction de traduction avec cache."""
    return translation_service.translate(text, lang)


# Initialisation des services
recommender: MovieRecommender = get_recommender()
metadata_service: MetadataService = get_metadata_service()

# Bouton de redirection
st.markdown(
    f"""
    <a href="{PORTFOLIO_URL}" target="_blank" style="text-decoration:none;">
    <div style="
    display: inline-block;
    background: linear-gradient(135deg, #6A11CB 0%, #2575FC 100%);
    color: white;
    padding: 12px 25px;
    border-radius: 30px;
    text-align: center;
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
    box-shadow: 0 4px 15px rgba(37, 117, 252, 0.3);
    transition: all 0.3s ease;
    text-transform: uppercase;
    letter-spacing: 1px;
    border: 2px solid transparent;
    position: relative;
    overflow: hidden;
    ">
    {_("Retour")}
    </div>
    </a>
    """,
    unsafe_allow_html=True
)

# --- Interface Streamlit ---
st.set_page_config(layout=APP_LAYOUT, page_title=_(APP_TITLE))
st.title(_("🎬 Ciné-Reco : Votre Guide Cinéma Personnalisé"))

if recommender.is_ready():
    # --- Barre latérale pour la notation ---
    st.sidebar.header(_("🔍 Notez des films"))
    
    if 'user_ratings' not in st.session_state:
        st.session_state.user_ratings = {}
    
    # Recherche HORS du formulaire
    movie_list: List[str] = recommender.get_movie_list()
    search_term: str = st.sidebar.text_input(_("Rechercher un film à noter :"))
    
    if search_term:
        filtered_movie_list: List[str] = [
            m for m in movie_list if search_term.lower() in m.lower()
        ]
    else:
        filtered_movie_list = movie_list[:1000]
    
    with st.sidebar.form("rating_form"):
        if filtered_movie_list:
            selected_movie_title: str = st.selectbox(_("Choisissez un film"), filtered_movie_list)
        else:
            st.warning(_("Aucun film trouvé pour cette recherche. Essaie avec un titre en anglais ou un film sorti avant 2024."))
            selected_movie_title = None
        
        rating: float = st.slider(_("Votre note"), 1.0, 5.0, 3.0, 0.5)
        submitted: bool = st.form_submit_button(_("Ajouter la note"))
        
        if submitted and selected_movie_title:
            movie_id = recommender.get_movie_id_by_title(selected_movie_title)
            if movie_id:
                st.session_state.user_ratings[movie_id] = rating
                st.success(_("Note ajoutée pour :") + f" {selected_movie_title}")
    
    if st.session_state.user_ratings:
        st.sidebar.subheader(_("Vos notes :"))
        for movie_id, rating in st.session_state.user_ratings.items():
            movie_title = recommender.get_movie_title_by_id(movie_id)
            if movie_title:
                st.sidebar.write(f"- {movie_title}: **{rating} / 5.0**")
        if st.sidebar.button(_("🗑️ Vider les notes")):
            st.session_state.user_ratings = {}
            st.rerun()
    
    # --- Affichage principal des recommandations ---
    st.header(_("🌟 Vos Recommandations Personnalisées"))
    if len(st.session_state.user_ratings) >= 3:
        with st.spinner(_("Nous préparons votre sélection personnalisée...")):
            recommendations_df: pd.DataFrame = recommender.generate_recommendations(
                st.session_state.user_ratings
            )
        
        # Filtrage par genre
        all_genres: Set[str] = set()
        for genres_str in recommendations_df['Genres'].dropna():
            if genres_str and genres_str != "(no genres listed)":
                for genre in str(genres_str).split('|'):
                    if genre.strip():
                        all_genres.add(genre.strip())
        
        sorted_genres: List[str] = sorted(list(all_genres))
        selected_genres: List[str] = st.multiselect(_("Filtrer par genre :"), sorted_genres)
        
        if selected_genres:
            def has_selected_genre(genres_str: str) -> bool:
                """Vérifie si un film contient au moins un des genres sélectionnés."""
                if pd.isna(genres_str) or not genres_str:
                    return False
                return any(g in str(genres_str) for g in selected_genres)
            
            filtered_df: pd.DataFrame = recommendations_df[
                recommendations_df['Genres'].apply(has_selected_genre)
            ]
        else:
            filtered_df = recommendations_df
        
        st.subheader(_("Top") + f" {min(20, len(filtered_df))} " + _("des films pour vous :"))
        
        # Récupérer les métadonnées de tous les films en parallèle
        top_movies = filtered_df.head(20)
        movie_titles = [row['Titre'] for _, row in top_movies.iterrows()]
        
        # Exécuter les appels asynchrones en parallèle
        # Streamlit exécute chaque script dans un nouveau contexte, donc asyncio.run() fonctionne
        movies_data_dict = asyncio.run(metadata_service.get_movies_data_batch(movie_titles))
        
        cols = st.columns(5)
        for i, (idx, row) in enumerate(top_movies.iterrows()):
            col = cols[i % 5]
            with col:
                movie_data = movies_data_dict.get(row['Titre'])
                
                if movie_data and movie_data.get("poster") and movie_data["poster"] != "N/A":
                    st.image(movie_data["poster"], caption=f"{row['Note Prédite']:.1f} ⭐")
                else:
                    st.image(str(NO_POSTER_IMAGE_PATH), caption=f"{row['Note Prédite']:.1f} ⭐")
                
                with st.expander(f"_{row['Titre']}_"):
                    st.write(f"**{_('Genres')} :** {movie_data['genre'] if movie_data else row['Genres']}")
                    st.write(f"**{_('Note prédite')} :** {row['Note Prédite']:.2f}")
                    if movie_data:
                        st.write(f"**{_('Acteurs')} :** {movie_data['actors']}")
                        st.write(f"**{_('Résumé')} :** {movie_data['plot']}")
                        st.write(f"**{_('Note IMDb')} :** {movie_data['rating']} ⭐")
                        st.write(f"**{_('Année')} :** {movie_data['year']}")
    
    else:
        st.info(_("""👋 Bienvenue !
Veuillez noter au moins 3 films dans la barre latérale pour débloquer vos recommandations.
Si on vous propose un film que vous avez déjà vu, il suffit de le noter pour qu'il ne vous soit plus proposé.
Si on vous propose de mauvais films, il suffit de leur mettre une mauvaise note."""))
    
    # Section Présentation
    st.header(_("Présentation"))
    st.markdown(_(
        """Ce projet vise à recommander des films en fonction des notes attribuées par les utilisateurs. À l'ère du numérique, 
        les algorithmes de recommandation sont omniprésents et jouent un rôle crucial dans nos choix quotidiens, en suggérant 
        du contenu aligné avec nos préférences.

**Domaines d'application :**
- **E-commerce & Marketing :** Suggestion de produits similaires aux achats précédents pour augmenter les conversions
- **Services client :** Recommandation de services adaptés aux besoins identifiés de l'utilisateur
- **Divertissement :** Proposition de films, séries ou musiques correspondant aux goûts de chacun
- **Analyse de tendances :** Identification de tendances émergentes basées sur les comportements collectifs
- **Ressources humaines :** Mise en relation de profils compatibles (recrutement, networking)
- **Éducation :** Parcours d'apprentissage personnalisés selon le niveau et les centres d'intérêt

**Données utilisées :**

Ce système s'appuie sur le jeu de données MovieLens, qui contient :
- **20 763 films** couvrant la période de 1874 à 2024
- **8 493 utilisateurs** actifs
- **2 864 752 notes** au total (échelle de 0.5 à 5 étoiles)
- Données à jour jusqu'au 1er mai 2024
- Multiples genres cinématographiques pour affiner les recommandations"""
    ))
    
    # Section Architecture du Modèle
    st.header(_("Architecture du Modèle"))
    st.markdown(_(
        """Notre système repose sur un **réseau de neurones siamois à deux branches**, une architecture particulièrement 
        adaptée à l'apprentissage de similarités entre entités hétérogènes.

**Structure du modèle :**

Le modèle est composé de deux sous-réseaux parallèles :

1. **Branche utilisateur :** Transforme le profil utilisateur (historique de notes, préférences de genres) 
en une représentation vectorielle dense

2. **Branche film :** Encode les caractéristiques des films (genres, popularité, patterns de notation) 
dans un espace latent commun

**Composants techniques :**
- **Couches denses successives** (256 → 128 → 64 neurones) pour l'extraction de features hiérarchiques
- **Activation GELU** : Fonction d'activation continue favorisant une meilleure propagation du gradient
- **Normalisation par batch** : Stabilise l'apprentissage et accélère la convergence
- **Dropout (30%)** : Prévient le surapprentissage en désactivant aléatoirement certains neurones
- **Régularisation L2 (1e-6)** : Pénalise les poids élevés pour favoriser la généralisation
- **Normalisation L2 finale** : Projette les embeddings sur une hypersphère unitaire pour des comparaisons stables

**Couche de fusion :**

Les vecteurs normalisés sont combinés via deux opérations complémentaires :
- **Différence absolue** : Capture la dissimilarité entre utilisateur et film
- **Produit élément par élément** : Modélise la similarité bilinéaire et les interactions fines

**Prédiction finale :**

Une couche Dense avec activation sigmoïde produit un score de compatibilité entre 0 et 1, 
facilement convertible en note prédite sur l'échelle 0.5-5 étoiles.

**Limitations connues :**
- Pas de prise en compte des données textuelles (synopsis, critiques)
- L'ordre chronologique des notes n'est pas exploité
- Les évolutions temporelles des préférences ne sont pas modélisées

Malgré ces limitations, le modèle offre des recommandations fiables et pertinentes."""
    ))
    st.image(str(ARCHITECTURE_IMAGE_PATH), caption=_("Architecture du modèle neuronal"), use_container_width=True)
    
    # Section Résultats
    st.header(_("Performances du Modèle"))
    st.markdown(_(
        """**Capacités du système :**

Le modèle entraîné permet deux types d'utilisation :
1. **Prédiction de note** : Estimer la note qu'un utilisateur attribuerait à un film non vu
2. **Recommandation personnalisée** : Suggérer les films avec les meilleures notes prédites pour un utilisateur donné

**Métriques de performance :**
- **RMSE (Root Mean Square Error) : 0.35** - Erreur moyenne de prédiction
- **MSE (Mean Square Error) : 0.12** - Métrique d'optimisation du modèle

Ces résultats sont satisfaisants pour un système de recommandation : une erreur de ~0.35 étoile 
représente une précision acceptable dans la prédiction des préférences cinématographiques.

À titre de comparaison, les systèmes de recommandation professionnels atteignent généralement des RMSE 
entre 0.25 et 0.40 sur MovieLens, positionnant notre modèle dans une fourchette compétitive."""
    ))
    
    # Section Coût et Maintenance
    st.header(_("Développement et Déploiement"))
    st.markdown(_(
        """**Infrastructure d'entraînement :**
- Matériel utilisé : MacBook M1 (sans GPU dédié)
- Temps de préparation des données : ~30 minutes
- Durée d'entraînement : 35 minutes
- Coût total : 0€ (aucune ressource cloud nécessaire)

**Caractéristiques du modèle en production :**
- Taille du modèle : 1.8 Mo (déploiement léger)
- Temps d'inférence : < 1 seconde pour générer des recommandations
- Scalabilité : Compatible avec des environnements à ressources limitées

**Coûts opérationnels :**
- **Entraînement** : Gratuit (CPU standard suffisant)
- **Hébergement** : Minimal (faible empreinte mémoire)
- **Maintenance** : Mise à jour périodique du dataset et réentraînement occasionnel

**Axes d'amélioration futurs :**
- Intégration de données textuelles (NLP sur synopsis et critiques)
- Prise en compte de la dimension temporelle (évolution des goûts)
- Ajout de features contextuelles (heure, dispositif, météo)
- Modèle hybride combinant filtrage collaboratif et approche content-based
- A/B testing pour optimiser les hyperparamètres en production
- Explainability : visualisation des facteurs influençant chaque recommandation"""
    ))

else:
    st.error(_("L'application n'a pas pu démarrer. Vérifiez les fichiers du modèle et des données."))

# Footer
st.markdown("---")
st.markdown(_(
    """
    Développé par [Gabriel Marie-Brisson](https://gabriel.mariebrisson.fr)
    """
))
