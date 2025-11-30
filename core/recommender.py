"""Module contenant la classe MovieRecommender pour la gestion du modèle et des recommandations."""

import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st

from config import (
    MODEL_PATH,
    SCALER_USER_PATH,
    SCALER_ITEM_PATH,
    SCALER_TARGET_PATH,
    MOVIE_DICT_PATH,
    ITEM_VECS_PATH,
    UNIQUE_GENRES_PATH,
)
from core.monitoring import (
    StructuredLogger,
    RecommendationMetrics,
    generate_trace_id,
    get_logger,
)


class MovieRecommender:
    """Classe responsable du chargement du modèle et de la génération de recommandations."""
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        scaler_user_path: Optional[Path] = None,
        scaler_item_path: Optional[Path] = None,
        scaler_target_path: Optional[Path] = None,
        movie_dict_path: Optional[Path] = None,
        item_vecs_path: Optional[Path] = None,
        unique_genres_path: Optional[Path] = None,
        logger: Optional[StructuredLogger] = None,
    ) -> None:
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
            logger: Logger structuré (optionnel, crée un logger par défaut si None)
        """
        self.model_path: Path = model_path or MODEL_PATH
        self.scaler_user_path: Path = scaler_user_path or SCALER_USER_PATH
        self.scaler_item_path: Path = scaler_item_path or SCALER_ITEM_PATH
        self.scaler_target_path: Path = scaler_target_path or SCALER_TARGET_PATH
        self.movie_dict_path: Path = movie_dict_path or MOVIE_DICT_PATH
        self.item_vecs_path: Path = item_vecs_path or ITEM_VECS_PATH
        self.unique_genres_path: Path = unique_genres_path or UNIQUE_GENRES_PATH
        
        self.model: Optional[tf.keras.Model] = None
        self.scaler_user: Optional[Any] = None
        self.scaler_item: Optional[Any] = None
        self.scaler_target: Optional[Any] = None
        self.movie_dict: Optional[Dict[int, Dict[str, Any]]] = None
        self.item_vecs_finder: Optional[np.ndarray] = None
        self.unique_genres: Optional[List[str]] = None
        
        # Logger pour observabilité
        self.logger: StructuredLogger = logger or get_logger()
        
        # Cache pour les embeddings utilisateur (amélioration performance)
        self._embedding_cache: Dict[str, np.ndarray] = {}
    
    @st.cache_resource
    def _load_model(self) -> Optional[tf.keras.Model]:
        """
        Charge le modèle Keras avec les objets personnalisés.
        
        Returns:
            Le modèle Keras chargé ou None en cas d'erreur
        """
        try:
            def l2_norm(x: tf.Tensor) -> tf.Tensor:
                """Normalisation L2."""
                return tf.linalg.l2_normalize(x, axis=1)

            def diff_abs(x: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
                """Différence absolue entre deux tenseurs."""
                return tf.abs(x[0] - x[1])

            def prod_mul(x: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
                """Produit élément par élément de deux tenseurs."""
                return x[0] * x[1]

            return tf.keras.models.load_model(
                str(self.model_path),
                custom_objects={
                    'l2_norm': l2_norm,
                    'diff_abs': diff_abs,
                    'prod_mul': prod_mul
                },
                safe_mode=False
            )
        except (OSError, IOError) as e:
            error_msg = f"Erreur lors du chargement du modèle (fichier non trouvé) : {e}"
            st.error(error_msg)
            # Logger aussi via structured logger si disponible
            if hasattr(self, 'logger'):
                self.logger.error(
                    "Failed to load model: file not found",
                    error=e,
                    model_path=str(self.model_path),
                )
            return None
        except (ValueError, AttributeError) as e:
            error_msg = f"Erreur lors du chargement du modèle (format invalide) : {e}"
            st.error(error_msg)
            if hasattr(self, 'logger'):
                self.logger.error(
                    "Failed to load model: invalid format",
                    error=e,
                    model_path=str(self.model_path),
                )
            return None
        except Exception as e:
            error_msg = f"Erreur inattendue lors du chargement du modèle : {e}"
            st.error(error_msg)
            if hasattr(self, 'logger'):
                self.logger.critical(
                    "Unexpected error loading model",
                    error=e,
                    model_path=str(self.model_path),
                    include_stacktrace=True,
                )
            return None
    
    @st.cache_data
    def _load_objects(
        self
    ) -> Tuple[
        Optional[Any],
        Optional[Any],
        Optional[Any],
        Optional[Dict[int, Dict[str, Any]]],
        Optional[np.ndarray],
        Optional[List[str]],
    ]:
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
            error_msg = f"Fichier .pkl manquant : {e}"
            st.error(error_msg)
            if hasattr(self, 'logger'):
                self.logger.error(
                    "Failed to load objects: file not found",
                    error=e,
                )
            return (None,) * 6
        except (OSError, IOError) as e:
            error_msg = f"Erreur lors de la lecture d'un fichier .pkl : {e}"
            st.error(error_msg)
            if hasattr(self, 'logger'):
                self.logger.error(
                    "Failed to load objects: I/O error",
                    error=e,
                )
            return (None,) * 6
        except (pickle.UnpicklingError, EOFError) as e:
            error_msg = f"Erreur lors du désérialisation d'un fichier .pkl : {e}"
            st.error(error_msg)
            if hasattr(self, 'logger'):
                self.logger.error(
                    "Failed to load objects: unpickling error",
                    error=e,
                )
            return (None,) * 6
    
    def initialize(self) -> bool:
        """
        Initialise le modèle et charge tous les objets nécessaires.
        
        Returns:
            True si l'initialisation a réussi, False sinon
        """
        self.model = self._load_model()
        (
            self.scaler_user,
            self.scaler_item,
            self.scaler_target,
            self.movie_dict,
            self.item_vecs_finder,
            self.unique_genres,
        ) = self._load_objects()
        
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
            self.unique_genres is not None,
        ])
    
    def generate_recommendations(
        self,
        user_ratings: Dict[int, float],
        trace_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Génère des recommandations de films basées sur les notes de l'utilisateur.
        
        Cette méthode est wrappée avec observabilité complète : logging structuré,
        métriques de performance, et gestion d'erreurs robuste.
        
        Args:
            user_ratings: Dictionnaire {movie_id: rating} des notes de l'utilisateur
            trace_id: ID de trace pour corrélation (généré automatiquement si None)
        
        Returns:
            DataFrame pandas trié par note prédite décroissante, contenant les colonnes:
            - Movie ID
            - Titre
            - Genres
            - Note Prédite
        
        Raises:
            RuntimeError: Si le modèle n'est pas prêt (avec log d'erreur)
            ValueError: Si les données d'entrée sont invalides (avec log d'erreur)
        """
        # Générer trace_id si non fourni
        if trace_id is None:
            trace_id = generate_trace_id()
        
        start_time = time.time()
        cache_hit = False
        
        try:
            # Validation précoce avec logging
            if not self.is_ready():
                error_msg = "Recommender not ready: model or data not loaded"
                self.logger.error(
                    error_msg,
                    trace_id=trace_id,
                    component="generate_recommendations",
                    reason="not_ready",
                )
                raise RuntimeError(error_msg)
            
            if not user_ratings:
                self.logger.warning(
                    "Empty user_ratings provided",
                    trace_id=trace_id,
                    component="generate_recommendations",
                )
                return pd.DataFrame()
            
            # Log de début de traitement
            self.logger.info(
                "Starting recommendation generation",
                trace_id=trace_id,
                component="generate_recommendations",
                num_ratings=len(user_ratings),
            )
            
            # Calcul des statistiques utilisateur
            num_ratings: int = len(user_ratings)
            avg_rating: float = float(np.mean(list(user_ratings.values())))
            
            # Calcul des préférences par genre
            genre_ratings: Dict[str, List[float]] = defaultdict(list)
            for movie_id, rating in user_ratings.items():
                if movie_id not in self.movie_dict:
                    self.logger.debug(
                        f"Movie ID {movie_id} not found in movie_dict",
                        trace_id=trace_id,
                        movie_id=movie_id,
                    )
                    continue
                genres_str: Union[str, float] = self.movie_dict[movie_id]['genres']
                if pd.notna(genres_str) and genres_str != "(no genres listed)":
                    for genre in str(genres_str).split('|'):
                        genre_ratings[genre].append(float(rating))
            
            # Construction du vecteur utilisateur
            user_prefs: Dict[str, float] = {
                f'pref_{g}': float(np.mean(genre_ratings.get(g, [avg_rating])))
                for g in self.unique_genres
            }
            user_vec: np.ndarray = np.array([[num_ratings, avg_rating, 0] + list(user_prefs.values())])
            
            # Préparation des données pour la prédiction
            num_items: int = len(self.item_vecs_finder)
            user_vecs_repeated: np.ndarray = np.tile(user_vec, (num_items, 1))
            
            # Normalisation avec gestion d'erreur
            try:
                suser_vecs: np.ndarray = self.scaler_user.transform(user_vecs_repeated)
                sitem_vecs: np.ndarray = self.scaler_item.transform(self.item_vecs_finder[:, 1:])
            except Exception as e:
                self.logger.error(
                    "Failed to transform features with scalers",
                    error=e,
                    trace_id=trace_id,
                    component="feature_scaling",
                    user_vec_shape=user_vec.shape,
                    item_vecs_shape=self.item_vecs_finder.shape,
                )
                raise RuntimeError(f"Feature scaling failed: {e}") from e
            
            # Prédiction avec gestion d'erreur
            try:
                predictions: np.ndarray = self.model.predict([suser_vecs, sitem_vecs], verbose=0)
                predictions_rescaled: np.ndarray = self.scaler_target.inverse_transform(predictions)
            except Exception as e:
                self.logger.error(
                    "Model prediction failed",
                    error=e,
                    trace_id=trace_id,
                    component="model_inference",
                    model_loaded=self.model is not None,
                    input_shapes={
                        "suser_vecs": suser_vecs.shape,
                        "sitem_vecs": sitem_vecs.shape,
                    },
                )
                raise RuntimeError(f"Model prediction failed: {e}") from e
            
            # Construction du DataFrame de recommandations
            recommendations: List[Dict[str, Union[int, str, float]]] = []
            for i, item_id in enumerate(self.item_vecs_finder[:, 0]):
                movie_id: int = int(item_id)
                if movie_id not in user_ratings and movie_id in self.movie_dict:
                    recommendations.append({
                        'Movie ID': movie_id,
                        'Titre': self.movie_dict[movie_id]['title'],
                        'Genres': self.movie_dict[movie_id]['genres'],
                        'Note Prédite': float(predictions_rescaled[i][0])
                    })
            
            reco_df: pd.DataFrame = pd.DataFrame(recommendations)
            if reco_df.empty:
                self.logger.warning(
                    "No recommendations generated",
                    trace_id=trace_id,
                    component="generate_recommendations",
                    num_items=num_items,
                    num_user_ratings=num_ratings,
                )
                return reco_df
            
            reco_df = reco_df.sort_values(by='Note Prédite', ascending=False)
            
            # Calcul des métriques
            inference_time_ms = (time.time() - start_time) * 1000
            
            metrics = RecommendationMetrics(
                trace_id=trace_id,
                num_input_ratings=num_ratings,
                num_recommendations=len(reco_df),
                inference_time_ms=inference_time_ms,
                cache_hit=cache_hit,
            )
            
            # Log des métriques
            self.logger.log_metrics(metrics, trace_id=trace_id)
            
            # Log de succès
            self.logger.info(
                "Recommendation generation completed successfully",
                trace_id=trace_id,
                component="generate_recommendations",
                num_recommendations=len(reco_df),
                inference_time_ms=inference_time_ms,
            )
            
            return reco_df
            
        except (RuntimeError, ValueError) as e:
            # Erreurs attendues (validation, état du système)
            inference_time_ms = (time.time() - start_time) * 1000
            
            metrics = RecommendationMetrics(
                trace_id=trace_id,
                num_input_ratings=len(user_ratings) if user_ratings else 0,
                num_recommendations=0,
                inference_time_ms=inference_time_ms,
                cache_hit=cache_hit,
                error=str(e),
                error_type=type(e).__name__,
            )
            
            self.logger.log_metrics(metrics, trace_id=trace_id)
            self.logger.error(
                "Recommendation generation failed with expected error",
                error=e,
                trace_id=trace_id,
                component="generate_recommendations",
                error_type=type(e).__name__,
            )
            
            # Re-raise pour que l'app puisse gérer (pas de silent failure)
            raise
            
        except Exception as e:
            # Erreurs inattendues (bugs, problèmes système)
            inference_time_ms = (time.time() - start_time) * 1000
            
            metrics = RecommendationMetrics(
                trace_id=trace_id,
                num_input_ratings=len(user_ratings) if user_ratings else 0,
                num_recommendations=0,
                inference_time_ms=inference_time_ms,
                cache_hit=cache_hit,
                error=str(e),
                error_type=type(e).__name__,
            )
            
            self.logger.log_metrics(metrics, trace_id=trace_id)
            self.logger.critical(
                "Recommendation generation failed with unexpected error",
                error=e,
                trace_id=trace_id,
                component="generate_recommendations",
                error_type=type(e).__name__,
                include_stacktrace=True,
            )
            
            # Re-raise pour éviter silent failure
            raise RuntimeError(
                f"Unexpected error in recommendation generation: {e}"
            ) from e
    
    def get_movie_list(self) -> List[str]:
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
