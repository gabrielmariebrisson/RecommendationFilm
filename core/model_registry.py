"""Module de gestion du registre de modèles (Model Registry)."""

import json
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import logging

import tensorflow as tf


@dataclass
class ModelMetadata:
    """Métadonnées d'un modèle versionné."""
    version: str
    model_path: Path
    accuracy: Optional[float] = None
    rmse: Optional[float] = None
    mse: Optional[float] = None
    training_date: Optional[str] = None
    commit_hash: Optional[str] = None
    description: Optional[str] = None
    is_stable: bool = True
    scaler_user_path: Optional[Path] = None
    scaler_item_path: Optional[Path] = None
    scaler_target_path: Optional[Path] = None
    movie_dict_path: Optional[Path] = None
    item_vecs_path: Optional[Path] = None
    unique_genres_path: Optional[Path] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit les métadonnées en dictionnaire."""
        result = asdict(self)
        # Convertir les Path en strings pour JSON
        for key, value in result.items():
            if isinstance(value, Path):
                result[key] = str(value)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Crée une instance depuis un dictionnaire."""
        # Convertir les strings en Path pour les chemins
        for key in ['model_path', 'scaler_user_path', 'scaler_item_path', 
                    'scaler_target_path', 'movie_dict_path', 'item_vecs_path', 
                    'unique_genres_path']:
            if key in data and data[key]:
                data[key] = Path(data[key])
        return cls(**data)


class ModelVersionManager:
    """Gestionnaire de versions de modèles avec validation et métadonnées."""
    
    def __init__(
        self,
        registry_path: Path,
        default_version: Optional[str] = None,
    ):
        """
        Initialise le gestionnaire de versions.
        
        Args:
            registry_path: Chemin vers le dossier du registre (ex: models/)
            default_version: Version par défaut à charger (None = dernière stable)
        """
        self.registry_path = Path(registry_path)
        self.default_version = default_version
        self.logger = logging.getLogger(__name__)
        
        # S'assurer que le dossier existe
        self.registry_path.mkdir(parents=True, exist_ok=True)
    
    def _get_git_commit_hash(self) -> Optional[str]:
        """
        Récupère le hash du commit Git actuel.
        
        Returns:
            Hash du commit ou None si Git n'est pas disponible
        """
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
            return None
    
    def register_model(
        self,
        model_path: Path,
        version: str,
        accuracy: Optional[float] = None,
        rmse: Optional[float] = None,
        mse: Optional[float] = None,
        description: Optional[str] = None,
        is_stable: bool = True,
        scaler_user_path: Optional[Path] = None,
        scaler_item_path: Optional[Path] = None,
        scaler_target_path: Optional[Path] = None,
        movie_dict_path: Optional[Path] = None,
        item_vecs_path: Optional[Path] = None,
        unique_genres_path: Optional[Path] = None,
    ) -> ModelMetadata:
        """
        Enregistre un nouveau modèle dans le registre.
        
        Args:
            model_path: Chemin vers le fichier .keras du modèle
            version: Version du modèle (ex: "v1", "v2")
            accuracy: Précision du modèle (optionnel)
            rmse: RMSE du modèle (optionnel)
            mse: MSE du modèle (optionnel)
            description: Description du modèle (optionnel)
            is_stable: Si True, marque comme version stable
            scaler_user_path: Chemin vers le scaler utilisateur
            scaler_item_path: Chemin vers le scaler item
            scaler_target_path: Chemin vers le scaler target
            movie_dict_path: Chemin vers le dictionnaire des films
            item_vecs_path: Chemin vers les vecteurs d'items
            unique_genres_path: Chemin vers les genres uniques
        
        Returns:
            ModelMetadata créé
        """
        version_dir = self.registry_path / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Copier le modèle dans le dossier versionné
        import shutil
        target_model_path = version_dir / "model.keras"
        shutil.copy2(model_path, target_model_path)
        
        # Créer les métadonnées
        metadata = ModelMetadata(
            version=version,
            model_path=target_model_path,
            accuracy=accuracy,
            rmse=rmse,
            mse=mse,
            training_date=datetime.now().isoformat(),
            commit_hash=self._get_git_commit_hash(),
            description=description,
            is_stable=is_stable,
            scaler_user_path=scaler_user_path,
            scaler_item_path=scaler_item_path,
            scaler_target_path=scaler_target_path,
            movie_dict_path=movie_dict_path,
            item_vecs_path=item_vecs_path,
            unique_genres_path=unique_genres_path,
        )
        
        # Sauvegarder les métadonnées
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Model version {version} registered successfully")
        return metadata
    
    def list_versions(self) -> List[str]:
        """
        Liste toutes les versions disponibles dans le registre.
        
        Returns:
            Liste des versions triées (plus récentes en premier)
        """
        versions = []
        for item in self.registry_path.iterdir():
            if item.is_dir() and (item / "metadata.json").exists():
                versions.append(item.name)
        
        # Trier par version (v1, v2, etc.)
        def version_key(v: str) -> int:
            try:
                return int(v.replace('v', ''))
            except ValueError:
                return 0
        
        return sorted(versions, key=version_key, reverse=True)
    
    def get_latest_stable_version(self) -> Optional[str]:
        """
        Retourne la dernière version stable.
        
        Returns:
            Version string ou None si aucune version stable trouvée
        """
        versions = self.list_versions()
        
        for version in versions:
            metadata = self.load_metadata(version)
            if metadata and metadata.is_stable:
                return version
        
        return None
    
    def load_metadata(self, version: Optional[str] = None) -> Optional[ModelMetadata]:
        """
        Charge les métadonnées d'une version.
        
        Args:
            version: Version à charger (None = dernière stable)
        
        Returns:
            ModelMetadata ou None si non trouvé
        """
        if version is None:
            version = self.default_version or self.get_latest_stable_version()
            if version is None:
                self.logger.warning("No stable version found")
                return None
        
        metadata_path = self.registry_path / version / "metadata.json"
        
        if not metadata_path.exists():
            self.logger.error(f"Metadata not found for version {version}")
            return None
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return ModelMetadata.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.logger.error(f"Failed to load metadata for version {version}: {e}")
            return None
    
    def validate_metadata(self, metadata: ModelMetadata) -> Tuple[bool, Optional[str]]:
        """
        Valide les métadonnées d'un modèle.
        
        Args:
            metadata: Métadonnées à valider
        
        Returns:
            Tuple (is_valid, error_message)
        """
        # Vérifier que le fichier modèle existe
        if not metadata.model_path.exists():
            return False, f"Model file not found: {metadata.model_path}"
        
        # Vérifier que les fichiers scalers existent si spécifiés
        scaler_paths = [
            metadata.scaler_user_path,
            metadata.scaler_item_path,
            metadata.scaler_target_path,
        ]
        for scaler_path in scaler_paths:
            if scaler_path and not scaler_path.exists():
                return False, f"Scaler file not found: {scaler_path}"
        
        # Vérifier les métadonnées critiques
        if metadata.version is None or not metadata.version:
            return False, "Version is required"
        
        if metadata.training_date is None:
            return False, "Training date is required"
        
        # Vérifier que le modèle n'est pas trop ancien (optionnel)
        # On peut ajouter une logique ici pour rejeter les modèles > X jours
        
        return True, None
    
    def load_model_paths(
        self,
        version: Optional[str] = None,
    ) -> Optional[Dict[str, Path]]:
        """
        Charge les chemins de tous les fichiers nécessaires pour une version.
        
        Args:
            version: Version à charger (None = dernière stable)
        
        Returns:
            Dictionnaire avec les chemins ou None si erreur
        """
        metadata = self.load_metadata(version)
        
        if metadata is None:
            return None
        
        # Valider les métadonnées
        is_valid, error = self.validate_metadata(metadata)
        if not is_valid:
            self.logger.error(f"Metadata validation failed: {error}")
            return None
        
        return {
            'model_path': metadata.model_path,
            'scaler_user_path': metadata.scaler_user_path,
            'scaler_item_path': metadata.scaler_item_path,
            'scaler_target_path': metadata.scaler_target_path,
            'movie_dict_path': metadata.movie_dict_path,
            'item_vecs_path': metadata.item_vecs_path,
            'unique_genres_path': metadata.unique_genres_path,
        }
    
    def get_model_info(self, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retourne les informations d'un modèle (métadonnées + statut).
        
        Args:
            version: Version à inspecter (None = dernière stable)
        
        Returns:
            Dictionnaire avec les informations du modèle
        """
        metadata = self.load_metadata(version)
        
        if metadata is None:
            return None
        
        is_valid, error = self.validate_metadata(metadata)
        
        return {
            'version': metadata.version,
            'is_stable': metadata.is_stable,
            'is_valid': is_valid,
            'validation_error': error,
            'accuracy': metadata.accuracy,
            'rmse': metadata.rmse,
            'mse': metadata.mse,
            'training_date': metadata.training_date,
            'commit_hash': metadata.commit_hash,
            'description': metadata.description,
            'model_path': str(metadata.model_path),
            'model_exists': metadata.model_path.exists() if metadata.model_path else False,
        }

