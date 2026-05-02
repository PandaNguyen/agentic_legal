from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


def compute_dynamic_centroids(
    vectors: list[list[float]],
    *,
    max_chunks_per_cluster: int = 20,
) -> tuple[list[list[float]], list[int]]:
    if not vectors:
        return [], []

    matrix = np.array(vectors, dtype=np.float32)
    if matrix.size == 0 or matrix.ndim != 2:
        return [], []

    normalized = normalize(matrix, norm="l2")
    cluster_size = max(1, int(max_chunks_per_cluster))
    cluster_count = math.ceil(normalized.shape[0] / cluster_size)

    if cluster_count <= 1:
        mean_vector = np.mean(normalized, axis=0)
        centroid = normalize(mean_vector.reshape(1, -1), norm="l2")
        return centroid.tolist(), [int(normalized.shape[0])]

    try:
        kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(normalized)
        centroids = normalize(kmeans.cluster_centers_, norm="l2")
        _, counts = np.unique(labels, return_counts=True)
        return centroids.tolist(), [int(count) for count in counts.tolist()]
    except Exception:
        mean_vector = np.mean(normalized, axis=0)
        centroid = normalize(mean_vector.reshape(1, -1), norm="l2")
        return centroid.tolist(), [int(normalized.shape[0])]


def max_cosine_similarity(query_vector: list[float], vectors: Iterable[list[float]]) -> float:
    query = np.array(query_vector, dtype=np.float32)
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0.0:
        return 0.0
    best = 0.0
    for vector in vectors:
        candidate = np.array(vector, dtype=np.float32)
        candidate_norm = float(np.linalg.norm(candidate))
        if candidate_norm == 0.0:
            continue
        score = float(np.dot(query, candidate) / (query_norm * candidate_norm))
        best = max(best, score)
    return best
