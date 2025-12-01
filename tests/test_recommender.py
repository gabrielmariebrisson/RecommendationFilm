"""Tests pour la classe MovieRecommender."""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any
import pandas as pd
import numpy as np

# Les mocks sont dans conftest.py
from src.core.recommender import MovieRecommender


class TestMovieRecommender:
    """Tests pour la classe MovieRecommender."""

    @pytest.fixture
    def mock_movie_dict(self):
        """Fixture pour un dictionnaire de films mocké."""
        return {
            1: {"title": "The Matrix", "genres": "Action|Sci-Fi"},
            2: {"title": "Inception", "genres": "Action|Sci-Fi|Thriller"},
            3: {"title": "Titanic", "genres": "Drama|Romance"},
            4: {"title": "The Dark Knight", "genres": "Action|Crime|Drama"},
            5: {"title": "Pulp Fiction", "genres": "Crime|Drama"},
        }

    @pytest.fixture
    def mock_unique_genres(self):
        """Fixture pour la liste des genres uniques."""
        return ["Action", "Sci-Fi", "Thriller", "Drama", "Romance", "Crime"]

    @pytest.fixture
    def mock_item_vecs(self):
        """Fixture pour les vecteurs d'items mockés."""
        # Créer un array numpy avec 5 items (movie_id, feature1, feature2, ...)
        return np.array(
            [
                [1, 0.5, 0.3, 0.2],
                [2, 0.6, 0.4, 0.3],
                [3, 0.4, 0.5, 0.1],
                [4, 0.7, 0.3, 0.4],
                [5, 0.5, 0.6, 0.2],
            ]
        )

    @pytest.fixture
    def mock_scalers(self):
        """Fixture pour les scalers mockés."""
        scaler_user = Mock()

        # Le scaler_user.transform reçoit un array de shape (num_items, num_features)
        # On mocke pour retourner un array de la même shape
        def mock_user_transform(x):
            return np.random.rand(x.shape[0], x.shape[1]).astype(np.float32)

        scaler_user.transform = Mock(side_effect=mock_user_transform)

        scaler_item = Mock()

        # Le scaler_item.transform reçoit item_vecs[:, 1:] donc shape (5, 3)
        def mock_item_transform(x):
            return np.random.rand(x.shape[0], x.shape[1]).astype(np.float32)

        scaler_item.transform = Mock(side_effect=mock_item_transform)

        scaler_target = Mock()

        # Le scaler_target.inverse_transform reçoit les prédictions
        def mock_target_inverse(x):
            # Retourner des notes prédites réalistes entre 0.5 et 5.0
            return np.random.uniform(3.0, 4.5, size=(x.shape[0], 1)).astype(np.float32)

        scaler_target.inverse_transform = Mock(side_effect=mock_target_inverse)

        return scaler_user, scaler_item, scaler_target

    @pytest.fixture
    def mock_model(self, mock_item_vecs):
        """Fixture pour le modèle Keras mocké."""
        model = Mock()
        # Le modèle.predict reçoit [suser_vecs, sitem_vecs] et retourne des prédictions
        # On utilise mock_item_vecs pour déterminer num_items car c'est la source de vérité
        expected_num_items = len(mock_item_vecs)

        def mock_predict(inputs, verbose=0):
            # inputs est une liste [suser_broadcasted, sitem_vecs]
            # suser_broadcasted a shape (num_items, num_features)
            # sitem_vecs a shape (num_items, num_features)
            # On utilise toujours expected_num_items pour garantir la cohérence
            num_items = expected_num_items
            # Vérification de sécurité : si inputs[1] existe et a une taille différente, utiliser celle-ci
            if isinstance(inputs, list) and len(inputs) >= 2:
                try:
                    actual_num_items = inputs[1].shape[0]
                    # Vérifier que actual_num_items est bien un nombre (int ou np.integer)
                    # et non un MagicMock
                    if (
                        isinstance(actual_num_items, (int, np.integer))
                        and actual_num_items > 0
                    ):
                        num_items = int(actual_num_items)
                except (AttributeError, IndexError, TypeError):
                    pass
            return np.random.uniform(0.5, 1.0, size=(num_items, 1)).astype(np.float32)

        model.predict = Mock(side_effect=mock_predict)
        return model

    def test_generate_recommendations_dataframe_structure(
        self,
        mock_movie_dict,
        mock_unique_genres,
        mock_item_vecs,
        mock_scalers,
        mock_model,
    ):
        """Test que generate_recommendations retourne un DataFrame avec les colonnes attendues."""
        # Arrange
        recommender = MovieRecommender()
        recommender.model = mock_model
        recommender.scaler_user, recommender.scaler_item, recommender.scaler_target = (
            mock_scalers
        )
        recommender.movie_dict = mock_movie_dict
        recommender.item_vecs_finder = mock_item_vecs
        recommender.unique_genres = mock_unique_genres

        user_ratings = {1: 4.5, 2: 5.0}  # L'utilisateur a noté les films 1 et 2

        # Act
        result = recommender.generate_recommendations(user_ratings)

        # Assert
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            # Vérifier les colonnes attendues
            expected_columns = ["Movie ID", "Titre", "Genres", "Note Prédite"]
            assert all(col in result.columns for col in expected_columns)

            # Vérifier que les films notés par l'utilisateur ne sont pas dans les recommandations
            recommended_movie_ids = [int(x) for x in result["Movie ID"].tolist()]
            assert 1 not in recommended_movie_ids
            assert 2 not in recommended_movie_ids

            # Vérifier que les autres films sont présents
            assert (
                3 in recommended_movie_ids
                or 4 in recommended_movie_ids
                or 5 in recommended_movie_ids
            )

    def test_generate_recommendations_sorted_by_rating(
        self,
        mock_movie_dict,
        mock_unique_genres,
        mock_item_vecs,
        mock_scalers,
        mock_model,
    ):
        """Test que les recommandations sont triées par note prédite décroissante."""
        # Arrange
        recommender = MovieRecommender()
        recommender.model = mock_model
        recommender.scaler_user, recommender.scaler_item, recommender.scaler_target = (
            mock_scalers
        )
        recommender.movie_dict = mock_movie_dict
        recommender.item_vecs_finder = mock_item_vecs
        recommender.unique_genres = mock_unique_genres

        user_ratings = {1: 4.5}

        # Act
        result = recommender.generate_recommendations(user_ratings)

        # Assert
        if not result.empty and len(result) > 1:
            ratings = [float(x) for x in result["Note Prédite"].tolist()]
            # Vérifier que les notes sont triées en ordre décroissant
            assert ratings == sorted(ratings, reverse=True)

    def test_generate_recommendations_empty_when_not_ready(self):
        """Test que generate_recommendations lève une RuntimeError si le recommender n'est pas prêt."""
        # Arrange
        recommender = MovieRecommender()
        # Ne pas initialiser les composants

        user_ratings = {1: 4.5}

        # Act & Assert
        with pytest.raises(RuntimeError, match="Recommender not ready"):
            recommender.generate_recommendations(user_ratings)

    def test_generate_recommendations_empty_ratings(
        self,
        mock_movie_dict,
        mock_unique_genres,
        mock_item_vecs,
        mock_scalers,
        mock_model,
    ):
        """Test avec des notes vides."""
        # Arrange
        recommender = MovieRecommender()
        recommender.model = mock_model
        recommender.scaler_user, recommender.scaler_item, recommender.scaler_target = (
            mock_scalers
        )
        recommender.movie_dict = mock_movie_dict
        recommender.item_vecs_finder = mock_item_vecs
        recommender.unique_genres = mock_unique_genres

        user_ratings = {}

        # Act
        result = recommender.generate_recommendations(user_ratings)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_generate_recommendations_movie_not_in_dict(
        self,
        mock_movie_dict,
        mock_unique_genres,
        mock_item_vecs,
        mock_scalers,
        mock_model,
    ):
        """Test avec un movie_id qui n'existe pas dans movie_dict."""
        # Arrange
        recommender = MovieRecommender()
        recommender.model = mock_model
        recommender.scaler_user, recommender.scaler_item, recommender.scaler_target = (
            mock_scalers
        )
        recommender.movie_dict = mock_movie_dict
        recommender.item_vecs_finder = mock_item_vecs
        recommender.unique_genres = mock_unique_genres

        user_ratings = {999: 4.5}  # ID qui n'existe pas

        # Act
        result = recommender.generate_recommendations(user_ratings)

        # Assert
        # Le système devrait gérer gracieusement ce cas
        assert isinstance(result, pd.DataFrame)

    def test_get_movie_list(self, mock_movie_dict):
        """Test de get_movie_list."""
        # Arrange
        recommender = MovieRecommender()
        recommender.movie_dict = mock_movie_dict

        # Act
        result = recommender.get_movie_list()

        # Assert
        assert isinstance(result, list)
        assert len(result) == 5
        assert result == sorted(result)  # Vérifier que c'est trié
        assert "The Matrix" in result
        assert "Inception" in result

    def test_get_movie_list_empty_dict(self):
        """Test de get_movie_list avec un dictionnaire vide."""
        # Arrange
        recommender = MovieRecommender()
        recommender.movie_dict = None

        # Act
        result = recommender.get_movie_list()

        # Assert
        assert isinstance(result, list)
        assert result == []

    def test_get_movie_id_by_title(self, mock_movie_dict):
        """Test de get_movie_id_by_title."""
        # Arrange
        recommender = MovieRecommender()
        recommender.movie_dict = mock_movie_dict

        # Act
        result = recommender.get_movie_id_by_title("The Matrix")

        # Assert
        assert result == 1

    def test_get_movie_id_by_title_not_found(self, mock_movie_dict):
        """Test de get_movie_id_by_title avec un titre inexistant."""
        # Arrange
        recommender = MovieRecommender()
        recommender.movie_dict = mock_movie_dict

        # Act
        result = recommender.get_movie_id_by_title("Film Inexistant")

        # Assert
        assert result is None

    def test_get_movie_title_by_id(self, mock_movie_dict):
        """Test de get_movie_title_by_id."""
        # Arrange
        recommender = MovieRecommender()
        recommender.movie_dict = mock_movie_dict

        # Act
        result = recommender.get_movie_title_by_id(1)

        # Assert
        assert result == "The Matrix"

    def test_get_movie_title_by_id_not_found(self, mock_movie_dict):
        """Test de get_movie_title_by_id avec un ID inexistant."""
        # Arrange
        recommender = MovieRecommender()
        recommender.movie_dict = mock_movie_dict

        # Act
        result = recommender.get_movie_title_by_id(999)

        # Assert
        assert result is None

    def test_is_ready_true(
        self,
        mock_movie_dict,
        mock_unique_genres,
        mock_item_vecs,
        mock_scalers,
        mock_model,
    ):
        """Test de is_ready quand tous les composants sont chargés."""
        # Arrange
        recommender = MovieRecommender()
        recommender.model = mock_model
        recommender.scaler_user, recommender.scaler_item, recommender.scaler_target = (
            mock_scalers
        )
        recommender.movie_dict = mock_movie_dict
        recommender.item_vecs_finder = mock_item_vecs
        recommender.unique_genres = mock_unique_genres

        # Act
        result = recommender.is_ready()

        # Assert
        assert result is True

    def test_is_ready_false(self):
        """Test de is_ready quand des composants manquent."""
        # Arrange
        recommender = MovieRecommender()
        # Ne pas initialiser les composants

        # Act
        result = recommender.is_ready()

        # Assert
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
