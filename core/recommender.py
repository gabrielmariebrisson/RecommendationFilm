"""Module contenant la classe MovieRecommender pour la gestion du modèle et des recommandations."""

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from typing import Dict, Tuple, Optional, Any
import streamlit as st


class MovieRecommender:
    """Classe responsable du chargement du modèle et de la génération de recommandations."""
    
    def __init__(self, model_path: str = "./templates/assets/film/best_model.keras",
                 scaler_user_path: str = "./templates/assets/film/scalerUser.pkl",
                 scaler_item_path: str = "./templates/assets/film/scalerItem.pkl",
                 scaler_target_path: str = "./templates/assets/film/scalerTarget.pkl",
                 movie_dict_path: str = "./templates/assets/film/movie_dict.pkl",
                 item_vecs_path: str = "./templates/assets/film/item_vecs_finder.pkl",
                 unique_genres_path: str = "./templates/assets/film/unique_genres.pkl"):
        """
        Initialise le MovieRecommender avec les chemins vers les fichiers du modèle.
        
        Args:
            model_path: Chemin vers le fichier .keras du modèle
            scaler_user_path: Chemin vers le scaler utilisateur
            scaler_item_path: Chemin vers le scaler item
            scaler_target_path: Chemin vers le scaler target
            movie_dict_path: Chemin vers le dictionnaire des films
            item_vecs_path: Chemin vers les vecteurs d'items
            unique_genres_path: Chemin vers la liste des genres uniques
        """
        self.model_path = model_path
        self.scaler_user_path = scaler_user_path
        self.scaler_item_path = scaler_item_path
        self.scaler_target_path = scaler_target_path
        self.movie_dict_path = movie_dict_path
        self.item_vecs_path = item_vecs_path
        self.unique_genres_path = unique_genres_path
        
        self.model: Optional[tf.keras.Model] = None
        self.scaler_user = None
        self.scaler_item = None
        self.scaler_target = None
        self.movie_dict: Optional[Dict[int, Dict[str, Any]]] = None
        self.item_vecs_finder = None
        self.unique_genres = None
    
    @st.cache_resource
    def _load_model(self) -> Optional[tf.keras.Model]:
        """
        Charge le modèle Keras avec les objets personnalisés.
        
        Returns:
            Le modèle Keras chargé ou None en cas d'erreur
        """
        try:
            def l2_norm(x):
                return tf.linalg.l2_normalize(x, axis=1)

            def diff_abs(x):
                return tf.abs(x[0] - x[1])

            def prod_mul(x):
                return x[0] * x[1]

            return tf.keras.models.load_model(
                self.model_path,
                custom_objects={
                    'l2_norm': l2_norm,
                    'diff_abs': diff_abs,
                    'prod_mul': prod_mul
                },
                safe_mode=False
            )
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle : {e}")
            return None
    
    @st.cache_data
    def _load_objects(self) -> Tuple[Any, Any, Any, Optional[Dict], Any, Any]:
        """
        Charge tous les objets nécessaires (scalers, dictionnaires, etc.).
        
        Returns:
            Tuple contenant (scalerUser, scalerItem, scalerTarget, movie_dict, item_vecs_finder, unique_genres)
        """
        try:
            with open(self.scaler_user_path, 'rb') as f:
                scaler_user = pickle.load(f)
            with open(self.scaler_item_path, 'rb') as f:
                scaler_item = pickle.load(f)
            with open(self.scaler_target_path, 'rb') as f:
                scaler_target = pickle.load(f)
            with open(self.movie_dict_path, 'rb') as f:
                movie_dict = pickle.load(f)
            with open(self.item_vecs_path, 'rb') as f:
                item_vecs_finder = pickle.load(f)
            with open(self.unique_genres_path, 'rb') as f:
                unique_genres = pickle.load(f)
            return scaler_user, scaler_item, scaler_target, movie_dict, item_vecs_finder, unique_genres
        except FileNotFoundError as e:
            st.error(f"Fichiers .pkl manquants : {e}")
            return (None,) * 6
    
    def initialize(self) -> bool:
        """
        Initialise le modèle et charge tous les objets nécessaires.
        
        Returns:
            True si l'initialisation a réussi, False sinon
        """
        self.model = self._load_model()
        self.scaler_user, self.scaler_item, self.scaler_target, \
        self.movie_dict, self.item_vecs_finder, self.unique_genres = self._load_objects()
        
        return self.is_ready()
    
    def is_ready(self) -> bool:
        """
        Vérifie si le recommender est prêt à générer des recommandations.
        
        Returns:
            True si tous les composants sont chargés, False sinon
        """
        return all([
            self.model is not None,
            self.scaler_user is not None,
            self.scaler_item is not None,
            self.scaler_target is not None,
            self.movie_dict is not None,
            self.item_vecs_finder is not None,
            self.unique_genres is not None
        ])
    
    def generate_recommendations(self, user_ratings: Dict[int, float]) -> pd.DataFrame:
        """
        Génère des recommandations de films basées sur les notes de l'utilisateur.
        
        Args:
            user_ratings: Dictionnaire {movie_id: rating} des notes de l'utilisateur
        
        Returns:
            DataFrame pandas trié par note prédite décroissante, contenant les colonnes:
            - Movie ID
            - Titre
            - Genres
            - Note Prédite
        """
        if not self.is_ready():
            return pd.DataFrame()
        
        # Calcul des statistiques utilisateur
        num_ratings = len(user_ratings)
        avg_rating = np.mean(list(user_ratings.values()))
        
        # Calcul des préférences par genre
        genre_ratings = defaultdict(list)
        for movie_id, rating in user_ratings.items():
            genres_str = self.movie_dict[movie_id]['genres']
            if pd.notna(genres_str) and genres_str != "(no genres listed)":
                for genre in genres_str.split('|'):
                    genre_ratings[genre].append(rating)
        
        # Construction du vecteur utilisateur
        user_prefs = {
            f'pref_{g}': np.mean(genre_ratings.get(g, [avg_rating])) 
            for g in self.unique_genres
        }
        user_vec = np.array([[num_ratings, avg_rating, 0] + list(user_prefs.values())])
        
        # Préparation des données pour la prédiction
        num_items = len(self.item_vecs_finder)
        user_vecs_repeated = np.tile(user_vec, (num_items, 1))
        
        # Normalisation
        suser_vecs = self.scaler_user.transform(user_vecs_repeated)
        sitem_vecs = self.scaler_item.transform(self.item_vecs_finder[:, 1:])
        
        # Prédiction
        predictions = self.model.predict([suser_vecs, sitem_vecs], verbose=0)
        predictions_rescaled = self.scaler_target.inverse_transform(predictions)
        
        # Construction du DataFrame de recommandations
        recommendations = []
        for i, item_id in enumerate(self.item_vecs_finder[:, 0]):
            if int(item_id) not in user_ratings:
                recommendations.append({
                    'Movie ID': int(item_id),
                    'Titre': self.movie_dict[int(item_id)]['title'],
                    'Genres': self.movie_dict[int(item_id)]['genres'],
                    'Note Prédite': predictions_rescaled[i][0]
                })
        
        reco_df = pd.DataFrame(recommendations)
        return reco_df.sort_values(by='Note Prédite', ascending=False)
    
    def get_movie_list(self) -> list:
        """
        Retourne la liste triée de tous les titres de films disponibles.
        
        Returns:
            Liste triée des titres de films
        """
        if not self.movie_dict:
            return []
        return sorted([info['title'] for info in self.movie_dict.values()])
    
    def get_movie_id_by_title(self, title: str) -> Optional[int]:
        """
        Retourne l'ID d'un film à partir de son titre.
        
        Args:
            title: Titre du film
        
        Returns:
            L'ID du film ou None si non trouvé
        """
        if not self.movie_dict:
            return None
        return next(
            (mid for mid, info in self.movie_dict.items() if info['title'] == title),
            None
        )
    
    def get_movie_title_by_id(self, movie_id: int) -> Optional[str]:
        """
        Retourne le titre d'un film à partir de son ID.
        
        Args:
            movie_id: ID du film
        
        Returns:
            Le titre du film ou None si non trouvé
        """
        if not self.movie_dict or movie_id not in self.movie_dict:
            return None
        return self.movie_dict[movie_id].get('title')

