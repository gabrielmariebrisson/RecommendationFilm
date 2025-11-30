"""Tests pour le service MetadataService."""

import pytest
from unittest.mock import Mock, patch
import requests

from services.metadata import MetadataService


class TestMetadataService:
    """Tests pour la classe MetadataService."""
    
    def test_get_movie_data_success(self):
        """Test de récupération réussie des métadonnées d'un film."""
        # Arrange
        mock_response_data = {
            "Response": "True",
            "Title": "The Matrix",
            "Year": "1999",
            "Genre": "Action, Sci-Fi",
            "Director": "Lana Wachowski, Lilly Wachowski",
            "Actors": "Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss",
            "Plot": "A computer hacker learns about the true nature of reality.",
            "imdbRating": "8.7",
            "imdbVotes": "1,234,567",
            "Poster": "https://example.com/poster.jpg"
        }
        
        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None
        
        service = MetadataService(api_key="test_api_key")
        
        # Act
        with patch('services.metadata.requests.get', return_value=mock_response):
            result = service.get_movie_data("The Matrix")
        
        # Assert
        assert result is not None
        assert result["title"] == "The Matrix"
        assert result["year"] == "1999"
        assert result["genre"] == "Action, Sci-Fi"
        assert result["director"] == "Lana Wachowski, Lilly Wachowski"
        assert result["actors"] == "Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss"
        assert result["plot"] == "A computer hacker learns about the true nature of reality."
        assert result["rating"] == "8.7"
        assert result["votes"] == "1,234,567"
        assert result["poster"] == "https://example.com/poster.jpg"
    
    def test_get_movie_data_with_year_in_title(self):
        """Test avec année dans le titre du film."""
        mock_response_data = {
            "Response": "True",
            "Title": "The Matrix",
            "Year": "1999",
            "Genre": "Action, Sci-Fi",
            "Director": "Lana Wachowski",
            "Actors": "Keanu Reeves",
            "Plot": "A computer hacker learns about reality.",
            "imdbRating": "8.7",
            "imdbVotes": "1,234,567",
            "Poster": "https://example.com/poster.jpg"
        }
        
        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None
        
        service = MetadataService(api_key="test_api_key")
        
        with patch('services.metadata.requests.get', return_value=mock_response) as mock_get:
            result = service.get_movie_data("The Matrix (1999)")
            
            # Vérifier que l'année a été extraite et ajoutée à l'URL
            call_args = mock_get.call_args
            url = call_args[0][0] if call_args[0] else str(call_args)
            assert "y=1999" in url or "y=1999" in str(call_args)
        
        assert result is not None
        assert result["title"] == "The Matrix"
    
    def test_get_movie_data_api_returns_false(self):
        """Test quand l'API retourne Response=False (film non trouvé)."""
        mock_response_data = {
            "Response": "False",
            "Error": "Movie not found!"
        }
        
        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None
        
        service = MetadataService(api_key="test_api_key")
        
        with patch('services.metadata.requests.get', return_value=mock_response):
            result = service.get_movie_data("Film Inexistant")
        
        assert result is None
    
    def test_get_movie_data_http_error_404(self):
        """Test de gestion d'une erreur HTTP 404."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        
        service = MetadataService(api_key="test_api_key")
        
        with patch('services.metadata.requests.get', return_value=mock_response):
            result = service.get_movie_data("Film Inexistant")
        
        assert result is None
    
    def test_get_movie_data_timeout(self):
        """Test de gestion d'un timeout."""
        service = MetadataService(api_key="test_api_key")
        
        with patch('services.metadata.requests.get', side_effect=requests.exceptions.Timeout("Timeout")):
            result = service.get_movie_data("The Matrix")
        
        assert result is None
    
    def test_get_movie_data_empty_title(self):
        """Test avec un titre vide."""
        service = MetadataService(api_key="test_api_key")
        
        mock_response_data = {
            "Response": "False",
            "Error": "Incorrect IMDb ID."
        }
        
        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None
        
        with patch('services.metadata.requests.get', return_value=mock_response):
            result = service.get_movie_data("")
        
        # L'API devrait retourner une erreur pour un titre vide
        assert result is None
    
    def test_get_movie_data_no_api_key(self):
        """Test sans clé API."""
        service = MetadataService(api_key=None)
        
        result = service.get_movie_data("The Matrix")
        
        assert result is None
    
    def test_get_movie_data_request_exception(self):
        """Test de gestion d'une exception de requête générique."""
        service = MetadataService(api_key="test_api_key")
        
        with patch('services.metadata.requests.get', side_effect=requests.exceptions.RequestException("Network error")):
            result = service.get_movie_data("The Matrix")
        
        assert result is None
    
    def test_get_movie_data_json_parsing_error(self):
        """Test de gestion d'une erreur de parsing JSON."""
        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.raise_for_status.return_value = None
        
        service = MetadataService(api_key="test_api_key")
        
        with patch('services.metadata.requests.get', return_value=mock_response):
            result = service.get_movie_data("The Matrix")
        
        assert result is None
    
    def test_get_movie_data_missing_fields(self):
        """Test avec des champs manquants dans la réponse API."""
        mock_response_data = {
            "Response": "True",
            "Title": "The Matrix",
            # Champs manquants : Year, Genre, etc.
        }
        
        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None
        
        service = MetadataService(api_key="test_api_key")
        
        with patch('services.metadata.requests.get', return_value=mock_response):
            result = service.get_movie_data("The Matrix")
        
        assert result is not None
        assert result["title"] == "The Matrix"
        assert result["year"] == "N/A"  # Valeur par défaut
        assert result["genre"] == "N/A"  # Valeur par défaut

