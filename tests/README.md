# Tests du système de recommandation de films

## Structure des tests

- `test_metadata.py` : Tests pour `MetadataService` (API OMDb)
- `test_recommender.py` : Tests pour `MovieRecommender` (logique de recommandation)
- `conftest.py` : Configuration pytest avec mocks des dépendances

## Exécution des tests

```bash
# Tous les tests
pytest tests/ -v

# Tests spécifiques
pytest tests/test_metadata.py -v
pytest tests/test_recommender.py -v

# Avec couverture
pytest tests/ --cov=core --cov=services --cov-report=html
```

## Tests implémentés

### MetadataService
- ✅ Récupération réussie des métadonnées
- ✅ Extraction de l'année depuis le titre
- ✅ Gestion des erreurs HTTP (404, timeout)
- ✅ Gestion des titres vides
- ✅ Gestion des réponses API invalides
- ✅ Gestion des champs manquants

### MovieRecommender
- ✅ Structure du DataFrame de recommandations
- ✅ Tri par note prédite
- ✅ Gestion des cas limites (pas prêt, notes vides)
- ✅ Méthodes utilitaires (get_movie_list, get_movie_id_by_title, etc.)
- ✅ Vérification de l'état (is_ready)

## Notes

Les tests utilisent `unittest.mock` pour mocker les appels réseau et éviter tout appel réel à l'API OMDb pendant les tests.

