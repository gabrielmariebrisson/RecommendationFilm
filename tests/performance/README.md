# Performance Benchmarks

Scripts de benchmark pour mesurer les performances du système de recommandation à l'échelle.

## Scripts

- `benchmark_scale.py`: Benchmark principal avec tests à 100, 1k, 10k utilisateurs
- `generate_report.py`: Génère le rapport Markdown `BENCHMARK_RESULTS.md`

## Utilisation

### Benchmark simple

```bash
cd tests/performance
python benchmark_scale.py
```

### Générer le rapport complet

```bash
cd tests/performance
python generate_report.py
```

Le rapport sera généré dans `BENCHMARK_RESULTS.md` à la racine du projet.

## Métriques mesurées

- **Latence d'inférence**: Temps moyen, P95, P99
- **Pic mémoire**: Utilisation mémoire maximale (tracemalloc)
- **Throughput**: Requêtes par seconde
- **Taux de succès**: Pourcentage de requêtes réussies

## Optimisations testées

- Broadcasting TensorFlow au lieu de `np.tile()` (réduction mémoire O(N) → O(1))
- Transformation unique du vecteur utilisateur (gain de ~20k×)

