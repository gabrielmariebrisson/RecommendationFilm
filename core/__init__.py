"""Core module for movie recommendation system."""

from .recommender import MovieRecommender
from .monitoring import (
    StructuredLogger,
    RecommendationMetrics,
    generate_trace_id,
    get_logger,
)

__all__ = [
    'MovieRecommender',
    'StructuredLogger',
    'RecommendationMetrics',
    'generate_trace_id',
    'get_logger',
]

