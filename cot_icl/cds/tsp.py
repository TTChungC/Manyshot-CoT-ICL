"""TSP-based demonstration ordering (CDS core)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm


def get_optimal_order(
    E,
    strategy: str = "cds",
    n_starts: int = 10,
):
    """
    Return (best_order, score) for embedding matrix E.

    strategy: cds | high_curvature
    """
    if hasattr(E, "numpy") and callable(E.numpy):
        E = E.numpy()
    if not isinstance(E, np.ndarray):
        E = np.asarray(E)

    X = E

    def create_distance_matrix(X_combined, strat="cds", curvature_weight=1.0, euclidean_weight=0.2):
        n = len(X_combined)
        euclidean_dist = squareform(pdist(X_combined, "euclidean"))
        euclidean_dist = euclidean_dist / (np.max(euclidean_dist) + 1e-8)
        curvature_dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                vec_ij = X_combined[j] - X_combined[i]
                norm_ij = np.linalg.norm(vec_ij)
                if norm_ij <= 0:
                    continue
                vec_ij = vec_ij / norm_ij
                midpoint = (X_combined[i] + X_combined[j]) / 2
                distances = np.linalg.norm(X_combined - midpoint, axis=1)
                distances[i], distances[j] = np.inf, np.inf
                k = np.argmin(distances)
                if k < n:
                    vec_jk = X_combined[k] - X_combined[j]
                    norm_jk = np.linalg.norm(vec_jk)
                    if norm_jk > 0:
                        vec_jk = vec_jk / norm_jk
                        cos_angle = np.clip(np.dot(vec_ij, vec_jk), -1, 1)
                        curvature_dist[i, j] = np.arccos(cos_angle) / np.pi
        if strat == "high_curvature":
            curvature_dist = np.max(curvature_dist) - curvature_dist
        return euclidean_weight * euclidean_dist + curvature_weight * curvature_dist

    def tsp_nearest_neighbor(dist_matrix, start_idx):
        n = len(dist_matrix)
        unvisited = set(range(n))
        unvisited.remove(start_idx)
        tour = [start_idx]
        current = start_idx
        while unvisited:
            nearest = min(unvisited, key=lambda x: dist_matrix[current, x])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        return tour

    def tsp_2opt(tour, dist_matrix, max_iterations=1000):
        n = len(tour)
        best_tour = tour.copy()
        best_distance = sum(dist_matrix[best_tour[i], best_tour[(i + 1) % n]] for i in range(n))
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            for i in range(1, n - 2):
                for j in range(i + 1, n):
                    if j - i == 1:
                        continue
                    new_tour = best_tour[:i] + best_tour[i : j + 1][::-1] + best_tour[j + 1 :]
                    new_distance = sum(
                        dist_matrix[new_tour[k], new_tour[(k + 1) % n]] for k in range(n)
                    )
                    if new_distance < best_distance:
                        best_tour, best_distance = new_tour, new_distance
                        improved = True
            iterations += 1
        return best_tour, best_distance

    def break_cycle_to_path(tour, dist_matrix):
        n = len(tour)
        longest_edge_idx, longest_edge_length = 0, 0
        for i in range(n):
            j = (i + 1) % n
            if dist_matrix[tour[i], tour[j]] > longest_edge_length:
                longest_edge_length = dist_matrix[tour[i], tour[j]]
                longest_edge_idx = i
        start_idx = (longest_edge_idx + 1) % n
        return [tour[(start_idx + i) % n] for i in range(n)]

    def compute_curvature_score(E_order):
        curvatures = []
        for i in range(1, len(E_order) - 1):
            v1 = E_order[i] - E_order[i - 1]
            v2 = E_order[i + 1] - E_order[i]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
                curvatures.append(np.arccos(cos_angle))
        return 1 / (1 + np.mean(curvatures)) if curvatures else 0

    if strategy not in ("cds", "high_curvature"):
        raise ValueError(f"Unknown strategy: {strategy}. Choose from: cds, high_curvature")

    print(f"Finding optimal ordering for {len(E)} embeddings (strategy={strategy})...")

    dist_matrix = create_distance_matrix(X, strategy=strategy)
    best_path = None
    best_path_score = -float("inf") if strategy == "cds" else float("inf")
    n_starts = min(n_starts, len(E))

    for start_idx in tqdm(range(n_starts), desc="Optimizing starts"):
        tour = tsp_nearest_neighbor(dist_matrix, start_idx)
        tour, _ = tsp_2opt(tour, dist_matrix, max_iterations=500)
        path = break_cycle_to_path(tour, dist_matrix)
        score = compute_curvature_score(X[path])
        if strategy == "cds" and score > best_path_score:
            best_path_score, best_path = score, path.copy()
        elif strategy == "high_curvature" and score < best_path_score:
            best_path_score, best_path = score, path.copy()

    print(f"Found ordering (strategy={strategy}) with score: {best_path_score:.4f}")
    return best_path, best_path_score


def get_best_order(E, strategy):
    order, _ = get_optimal_order(E, strategy)
    return order
