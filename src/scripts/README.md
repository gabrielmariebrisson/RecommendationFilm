# Scripts Utilitaires

## register_model.py

Enregistre un modèle dans le Model Registry.

### Utilisation

```bash
# Enregistrer le modèle actuel comme v1
python scripts/register_model.py v1 --rmse 0.85 --accuracy 0.92

# Enregistrer avec description
python scripts/register_model.py v2 \
    --rmse 0.82 \
    --accuracy 0.94 \
    --description "Modèle amélioré avec plus de données"

# Enregistrer comme version instable (beta)
python scripts/register_model.py v2-beta --unstable

# Spécifier des chemins personnalisés
python scripts/register_model.py v1 \
    --model-path /path/to/model.keras \
    --scaler-user /path/to/scaler.pkl
```

### Arguments

- `version`: Version du modèle (requis, ex: v1, v2)
- `--model-path`: Chemin vers le fichier .keras (défaut: config.MODEL_PATH)
- `--accuracy`: Précision du modèle (optionnel)
- `--rmse`: RMSE du modèle (optionnel)
- `--mse`: MSE du modèle (optionnel)
- `--description`: Description du modèle (optionnel)
- `--unstable`: Marquer comme instable (non-stable)
- `--scaler-user`, `--scaler-item`, `--scaler-target`: Chemins vers les scalers
- `--movie-dict`, `--item-vecs`, `--unique-genres`: Chemins vers les données

