"""Script de benchmark pour mesurer les performances à l'échelle."""

import asyncio
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

import numpy as np
import pandas as pd

# Ajouter le répertoire parent au path pour les imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.recommender import MovieRecommender
from core.monitoring import generate_trace_id


class BenchmarkResults:
    """Classe pour stocker et analyser les résultats de benchmark."""
    
    def __init__(self):
        self.results: List[Dict] = []
    
    def add_result(
        self,
        num_users: int,
        inference_time_ms: float,
        peak_memory_mb: float,
        success: bool,
        error: str = None,
    ):
        """Ajoute un résultat de benchmark."""
        self.results.append({
            "num_users": num_users,
            "inference_time_ms": inference_time_ms,
            "peak_memory_mb": peak_memory_mb,
            "success": success,
            "error": error,
        })
    
    def get_summary(self) -> Dict:
        """Retourne un résumé statistique des résultats."""
        if not self.results:
            return {}
        
        successful = [r for r in self.results if r["success"]]
        
        if not successful:
            return {"error": "No successful runs"}
        
        inference_times = [r["inference_time_ms"] for r in successful]
        memory_peaks = [r["peak_memory_mb"] for r in successful]
        
        return {
            "total_runs": len(self.results),
            "successful_runs": len(successful),
            "failed_runs": len(self.results) - len(successful),
            "inference_time": {
                "mean_ms": statistics.mean(inference_times),
                "median_ms": statistics.median(inference_times),
                "p95_ms": self._percentile(inference_times, 95),
                "p99_ms": self._percentile(inference_times, 99),
                "min_ms": min(inference_times),
                "max_ms": max(inference_times),
            },
            "memory": {
                "mean_mb": statistics.mean(memory_peaks),
                "median_mb": statistics.median(memory_peaks),
                "p95_mb": self._percentile(memory_peaks, 95),
                "max_mb": max(memory_peaks),
            },
        }
    
    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        """Calcule un percentile."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


def generate_mock_user_ratings(
    num_ratings: int = 5,
    movie_ids: List[int] = None,
    seed: int = None,
) -> Dict[int, float]:
    """
    Génère des notes utilisateur mockées pour les tests.
    
    Args:
        num_ratings: Nombre de notes à générer
        movie_ids: Liste des IDs de films disponibles
        seed: Seed pour la reproductibilité
    
    Returns:
        Dictionnaire {movie_id: rating}
    """
    import random
    if seed is not None:
        random.seed(seed)
    
    if movie_ids is None:
        # IDs mockés si pas de movie_dict disponible
        movie_ids = list(range(1, 1000))
    
    selected_movies = random.sample(movie_ids, min(num_ratings, len(movie_ids)))
    ratings = {mid: random.uniform(0.5, 5.0) for mid in selected_movies}
    return ratings


def benchmark_single_user(
    recommender: MovieRecommender,
    user_ratings: Dict[int, float],
) -> Tuple[float, float, bool, str]:
    """
    Benchmark une seule requête utilisateur.
    
    Returns:
        Tuple (inference_time_ms, peak_memory_mb, success, error_message)
    """
    tracemalloc.start()
    start_time = time.time()
    error_msg = None
    success = False
    
    try:
        trace_id = generate_trace_id()
        recommendations = recommender.generate_recommendations(
            user_ratings,
            trace_id=trace_id,
        )
        
        inference_time_ms = (time.time() - start_time) * 1000
        current, peak = tracemalloc.get_traced_memory()
        peak_memory_mb = peak / (1024 * 1024)  # Convertir en MB
        tracemalloc.stop()
        
        success = True
        return inference_time_ms, peak_memory_mb, success, error_msg
        
    except Exception as e:
        inference_time_ms = (time.time() - start_time) * 1000
        current, peak = tracemalloc.get_traced_memory()
        peak_memory_mb = peak / (1024 * 1024)
        tracemalloc.stop()
        
        error_msg = str(e)
        return inference_time_ms, peak_memory_mb, success, error_msg


def benchmark_concurrent_users(
    recommender: MovieRecommender,
    num_users: int,
    max_workers: int = 10,
) -> BenchmarkResults:
    """
    Benchmark plusieurs utilisateurs concurrents.
    
    Args:
        recommender: Instance de MovieRecommender
        num_users: Nombre d'utilisateurs simultanés
        max_workers: Nombre de threads parallèles
    
    Returns:
        BenchmarkResults avec tous les résultats
    """
    results = BenchmarkResults()
    
    # Générer des user_ratings mockées avec seeds pour reproductibilité
    movie_ids = list(recommender.movie_dict.keys()) if recommender.movie_dict else None
    user_ratings_list = [
        generate_mock_user_ratings(movie_ids=movie_ids, seed=i)
        for i in range(num_users)
    ]
    
    # Exécuter les requêtes en parallèle
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(benchmark_single_user, recommender, ratings)
            for ratings in user_ratings_list
        ]
        
        for future in as_completed(futures):
            inference_time, peak_memory, success, error = future.result()
            results.add_result(
                num_users=1,  # Par utilisateur
                inference_time_ms=inference_time,
                peak_memory_mb=peak_memory,
                success=success,
                error=error,
            )
    
    return results


def run_benchmark_suite(recommender: MovieRecommender) -> Dict:
    """
    Exécute une suite complète de benchmarks.
    
    Args:
        recommender: Instance de MovieRecommender initialisée
    
    Returns:
        Dictionnaire avec tous les résultats
    """
    print("🚀 Starting benchmark suite...")
    print("=" * 60)
    
    all_results = {}
    test_scales = [100, 1000, 10000]
    
    for num_users in test_scales:
        print(f"\n📊 Benchmarking {num_users} concurrent users...")
        print("-" * 60)
        
        start_time = time.time()
        results = benchmark_concurrent_users(
            recommender,
            num_users=num_users,
            max_workers=min(50, num_users),  # Limiter les workers
        )
        total_time = time.time() - start_time
        
        summary = results.get_summary()
        summary["total_wall_time_seconds"] = total_time
        summary["throughput_req_per_sec"] = num_users / total_time if total_time > 0 else 0
        
        all_results[f"{num_users}_users"] = summary
        
        print(f"✅ Completed {num_users} users in {total_time:.2f}s")
        if "inference_time" in summary:
            print(f"   Mean inference: {summary['inference_time']['mean_ms']:.2f}ms")
            print(f"   P95 inference: {summary['inference_time']['p95_ms']:.2f}ms")
            print(f"   Peak memory: {summary['memory']['max_mb']:.2f}MB")
            print(f"   Throughput: {summary['throughput_req_per_sec']:.2f} req/s")
    
    return all_results


def estimate_daily_capacity(results: Dict) -> Dict:
    """
    Estime la capacité quotidienne basée sur les résultats de benchmark.
    
    Args:
        results: Résultats de benchmark
    
    Returns:
        Estimation de capacité
    """
    # Utiliser les résultats de 1000 utilisateurs comme baseline
    baseline = results.get("1000_users", {})
    
    if not baseline or "throughput_req_per_sec" not in baseline:
        return {"error": "Insufficient benchmark data"}
    
    throughput = baseline["throughput_req_per_sec"]
    
    # Calculs de capacité
    seconds_per_day = 86400
    requests_per_day = throughput * seconds_per_day
    
    # Estimation conservative (70% de la capacité théorique pour marge de sécurité)
    conservative_capacity = int(requests_per_day * 0.7)
    
    # Estimation optimiste (90% de la capacité théorique)
    optimistic_capacity = int(requests_per_day * 0.9)
    
    return {
        "baseline_throughput_req_per_sec": throughput,
        "theoretical_max_per_day": int(requests_per_day),
        "conservative_estimate_per_day": conservative_capacity,
        "optimistic_estimate_per_day": optimistic_capacity,
        "can_handle_1m_per_day": conservative_capacity >= 1_000_000,
        "can_handle_10m_per_day": conservative_capacity >= 10_000_000,
    }


def main():
    """Fonction principale du benchmark."""
    print("=" * 60)
    print("🎬 Movie Recommendation System - Performance Benchmark")
    print("=" * 60)
    
    # Initialiser le recommender
    print("\n📦 Initializing MovieRecommender...")
    recommender = MovieRecommender()
    
    if not recommender.initialize():
        print("❌ Failed to initialize recommender. Check model files.")
        return
    
    if not recommender.is_ready():
        print("❌ Recommender not ready. Check model files.")
        return
    
    print("✅ Recommender initialized successfully")
    print(f"   Movies in catalog: {len(recommender.movie_dict) if recommender.movie_dict else 0}")
    
    # Exécuter les benchmarks
    results = run_benchmark_suite(recommender)
    
    # Estimer la capacité quotidienne
    capacity = estimate_daily_capacity(results)
    
    # Afficher les résultats
    print("\n" + "=" * 60)
    print("📈 BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    
    for scale, summary in results.items():
        print(f"\n{scale.upper()}:")
        if "inference_time" in summary:
            print(f"  Mean inference time: {summary['inference_time']['mean_ms']:.2f}ms")
            print(f"  P95 inference time: {summary['inference_time']['p95_ms']:.2f}ms")
            print(f"  Peak memory: {summary['memory']['max_mb']:.2f}MB")
            print(f"  Throughput: {summary['throughput_req_per_sec']:.2f} req/s")
            print(f"  Success rate: {summary['successful_runs']}/{summary['total_runs']}")
    
    print("\n" + "=" * 60)
    print("📊 DAILY CAPACITY ESTIMATION")
    print("=" * 60)
    if "error" not in capacity:
        print(f"  Baseline throughput: {capacity['baseline_throughput_req_per_sec']:.2f} req/s")
        print(f"  Theoretical max: {capacity['theoretical_max_per_day']:,} req/day")
        print(f"  Conservative estimate: {capacity['conservative_estimate_per_day']:,} req/day")
        print(f"  Optimistic estimate: {capacity['optimistic_estimate_per_day']:,} req/day")
        print(f"  Can handle 1M+/day: {'✅ YES' if capacity['can_handle_1m_per_day'] else '❌ NO'}")
        print(f"  Can handle 10M+/day: {'✅ YES' if capacity['can_handle_10m_per_day'] else '❌ NO'}")
    
    # Sauvegarder les résultats pour le rapport
    return {
        "benchmark_results": results,
        "capacity_estimation": capacity,
    }


if __name__ == "__main__":
    results = main()

