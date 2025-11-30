"""Core module for movie recommendation system."""

from .recommender import MovieRecommender
from .monitoring import (
    StructuredLogger,
    RecommendationMetrics,
    generate_trace_id,
    get_logger,
)
from .model_registry import (
    ModelVersionManager,
    ModelMetadata,
)

__all__ = [
    'MovieRecommender',
    'StructuredLogger',
    'RecommendationMetrics',
    'generate_trace_id',
    'get_logger',
    'ModelVersionManager',
    'ModelMetadata',
]

