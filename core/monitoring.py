"""Module de monitoring et observabilité pour le système de recommandation."""

import json
import logging
import traceback
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from enum import Enum


class LogLevel(str, Enum):
    """Niveaux de log standardisés."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class RecommendationMetrics:
    """Métriques pour une requête de recommandation."""
    trace_id: str
    num_input_ratings: int
    num_recommendations: int
    inference_time_ms: float
    cache_hit: bool
    error: Optional[str] = None
    error_type: Optional[str] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        """Génère le timestamp si non fourni."""
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit les métriques en dictionnaire."""
        return asdict(self)


class JSONFormatter(logging.Formatter):
    """Formateur de logs au format JSON structuré."""
    
    def __init__(self, service_name: str = "movie-recommender"):
        """
        Initialise le formateur JSON.
        
        Args:
            service_name: Nom du service pour identification
        """
        super().__init__()
        self.service_name = service_name
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Formate un log record en JSON.
        
        Args:
            record: LogRecord à formater
        
        Returns:
            Chaîne JSON formatée
        """
        log_entry = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "service": self.service_name,
            "message": record.getMessage(),
        }
        
        # Ajouter trace_id si présent
        if hasattr(record, "trace_id"):
            log_entry["trace_id"] = record.trace_id
        
        # Ajouter des champs supplémentaires depuis extra
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        
        # Ajouter exception info si présente
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "stacktrace": traceback.format_exception(*record.exc_info),
            }
        
        # Ajouter stacktrace même sans exception si demandé
        if hasattr(record, "include_stacktrace") and record.include_stacktrace:
            log_entry["stacktrace"] = traceback.format_stack()
        
        return json.dumps(log_entry, ensure_ascii=False)


class StructuredLogger:
    """Logger structuré pour production avec support JSON et trace IDs."""
    
    def __init__(
        self,
        name: str = "movie-recommender",
        service_name: str = "movie-recommender",
        level: int = logging.INFO,
    ):
        """
        Initialise le logger structuré.
        
        Args:
            name: Nom du logger
            service_name: Nom du service pour les logs
            level: Niveau de log (logging.INFO, logging.DEBUG, etc.)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.service_name = service_name
        
        # Éviter les handlers duplicatés
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(JSONFormatter(service_name=service_name))
            self.logger.addHandler(handler)
    
    def _log(
        self,
        level: int,
        message: str,
        trace_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Méthode interne pour logger avec trace_id et champs supplémentaires.
        
        Args:
            level: Niveau de log (logging.INFO, etc.)
            message: Message à logger
            trace_id: ID de trace pour corrélation
            **kwargs: Champs supplémentaires à inclure dans le log
        """
        extra = {"trace_id": trace_id} if trace_id else {}
        if kwargs:
            extra["extra_fields"] = kwargs
        
        self.logger.log(level, message, extra=extra)
    
    def debug(
        self,
        message: str,
        trace_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log au niveau DEBUG."""
        self._log(logging.DEBUG, message, trace_id, **kwargs)
    
    def info(
        self,
        message: str,
        trace_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log au niveau INFO."""
        self._log(logging.INFO, message, trace_id, **kwargs)
    
    def warning(
        self,
        message: str,
        trace_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log au niveau WARNING."""
        self._log(logging.WARNING, message, trace_id, **kwargs)
    
    def error(
        self,
        message: str,
        error: Optional[Exception] = None,
        trace_id: Optional[str] = None,
        include_stacktrace: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Log au niveau ERROR avec gestion d'exception.
        
        Args:
            message: Message d'erreur
            error: Exception à logger (optionnel)
            trace_id: ID de trace pour corrélation
            include_stacktrace: Inclure la stacktrace même sans exception
            **kwargs: Champs supplémentaires
        """
        extra = {"trace_id": trace_id} if trace_id else {}
        if kwargs:
            extra["extra_fields"] = kwargs
        if include_stacktrace:
            extra["include_stacktrace"] = True
        
        if error:
            self.logger.error(
                message,
                exc_info=(type(error), error, error.__traceback__),
                extra=extra,
            )
        else:
            self.logger.error(message, extra=extra)
    
    def critical(
        self,
        message: str,
        error: Optional[Exception] = None,
        trace_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Log au niveau CRITICAL avec gestion d'exception.
        
        Args:
            message: Message critique
            error: Exception à logger (optionnel)
            trace_id: ID de trace pour corrélation
            **kwargs: Champs supplémentaires
        """
        extra = {"trace_id": trace_id} if trace_id else {}
        if kwargs:
            extra["extra_fields"] = kwargs
        
        if error:
            self.logger.critical(
                message,
                exc_info=(type(error), error, error.__traceback__),
                extra=extra,
            )
        else:
            self.logger.critical(message, extra=extra)
    
    def log_metrics(
        self,
        metrics: RecommendationMetrics,
        trace_id: Optional[str] = None,
    ) -> None:
        """
        Log les métriques de recommandation.
        
        Args:
            metrics: Métriques à logger
            trace_id: ID de trace (utilise celui des métriques si None)
        """
        trace_id = trace_id or metrics.trace_id
        level = logging.ERROR if metrics.error else logging.INFO
        
        self._log(
            level,
            "recommendation_metrics",
            trace_id=trace_id,
            **metrics.to_dict(),
        )


def generate_trace_id() -> str:
    """
    Génère un ID de trace unique.
    
    Returns:
        UUID4 formaté en string
    """
    return str(uuid.uuid4())


# Instance globale du logger (peut être remplacée pour les tests)
_default_logger: Optional[StructuredLogger] = None


def get_logger(
    name: str = "movie-recommender",
    service_name: str = "movie-recommender",
) -> StructuredLogger:
    """
    Obtient ou crée l'instance globale du logger.
    
    Args:
        name: Nom du logger
        service_name: Nom du service
    
    Returns:
        Instance de StructuredLogger
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = StructuredLogger(name=name, service_name=service_name)
    return _default_logger

