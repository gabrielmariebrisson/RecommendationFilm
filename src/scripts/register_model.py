"""Script utilitaire pour enregistrer un modèle dans le Model Registry."""

import argparse
import sys
from pathlib import Path

# Ajouter le répertoire parent de src/ au path.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.model_registry import ModelVersionManager
from src.config import (
    MODEL_REGISTRY_PATH,
    MODEL_PATH,
    SCALER_USER_PATH,
    SCALER_ITEM_PATH,
    SCALER_TARGET_PATH,
    MOVIE_DICT_PATH,
    ITEM_VECS_PATH,
    UNIQUE_GENRES_PATH,
)


def main():
    """Enregistre le modèle actuel dans le registre."""
    parser = argparse.ArgumentParser(
        description="Enregistre un modèle dans le Model Registry"
    )
    parser.add_argument(
        'version',
        type=str,
        help='Version du modèle (ex: v1, v2)',
    )
    parser.add_argument(
        '--model-path',
        type=Path,
        default=MODEL_PATH,
        help='Chemin vers le fichier .keras du modèle',
    )
    parser.add_argument(
        '--accuracy',
        type=float,
        help='Précision du modèle (optionnel)',
    )
    parser.add_argument(
        '--rmse',
        type=float,
        help='RMSE du modèle (optionnel)',
    )
    parser.add_argument(
        '--mse',
        type=float,
        help='MSE du modèle (optionnel)',
    )
    parser.add_argument(
        '--description',
        type=str,
        help='Description du modèle (optionnel)',
    )
    parser.add_argument(
        '--unstable',
        action='store_true',
        help='Marquer le modèle comme instable (non-stable)',
    )
    parser.add_argument(
        '--scaler-user',
        type=Path,
        default=SCALER_USER_PATH,
        help='Chemin vers le scaler utilisateur',
    )
    parser.add_argument(
        '--scaler-item',
        type=Path,
        default=SCALER_ITEM_PATH,
        help='Chemin vers le scaler item',
    )
    parser.add_argument(
        '--scaler-target',
        type=Path,
        default=SCALER_TARGET_PATH,
        help='Chemin vers le scaler target',
    )
    parser.add_argument(
        '--movie-dict',
        type=Path,
        default=MOVIE_DICT_PATH,
        help='Chemin vers le dictionnaire des films',
    )
    parser.add_argument(
        '--item-vecs',
        type=Path,
        default=ITEM_VECS_PATH,
        help='Chemin vers les vecteurs d\'items',
    )
    parser.add_argument(
        '--unique-genres',
        type=Path,
        default=UNIQUE_GENRES_PATH,
        help='Chemin vers les genres uniques',
    )
    
    args = parser.parse_args()
    
    # Vérifier que le fichier modèle existe.
    if not args.model_path.exists():
        print(f"❌ Erreur: Le fichier modèle n'existe pas: {args.model_path}")
        return 1
    
    # Créer le gestionnaire de registre.
    registry = ModelVersionManager(registry_path=MODEL_REGISTRY_PATH)
    
    # Enregistrer le modèle.
    try:
        metadata = registry.register_model(
            model_path=args.model_path,
            version=args.version,
            accuracy=args.accuracy,
            rmse=args.rmse,
            mse=args.mse,
            description=args.description,
            is_stable=not args.unstable,
            scaler_user_path=args.scaler_user,
            scaler_item_path=args.scaler_item,
            scaler_target_path=args.scaler_target,
            movie_dict_path=args.movie_dict,
            item_vecs_path=args.item_vecs,
            unique_genres_path=args.unique_genres,
        )
        
        print(f"✅ Modèle version {args.version} enregistré avec succès!")
        print(f"   Chemin: {metadata.model_path}")
        print(f"   Date d'entraînement: {metadata.training_date}")
        if metadata.commit_hash:
            print(f"   Commit Git: {metadata.commit_hash}")
        if metadata.rmse:
            print(f"   RMSE: {metadata.rmse:.4f}")
        if metadata.accuracy:
            print(f"   Accuracy: {metadata.accuracy:.4f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Erreur lors de l'enregistrement: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

