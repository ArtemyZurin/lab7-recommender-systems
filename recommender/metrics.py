import numpy as np


def ranking_metrics(recommendations, relevant, k):
    """Binary relevance; AP@K denominator min(K, number of relevant items)."""
    if k < 1:
        raise ValueError('K must be positive')
    relevant = set(relevant)
    if not relevant:
        raise ValueError('Ranking metrics require a nonempty relevance set')
    recs = list(recommendations)[:k]
    if len(recs) != len(set(recs)):
        raise ValueError('Duplicate recommendations')
    hits = np.array([int(i in relevant) for i in recs], dtype=float)
    ranks = np.arange(1, len(hits) + 1)
    denom = min(k, len(relevant))
    return {'precision': float(hits.sum() / k), 'recall': float(hits.sum() / len(relevant)),
            'map': float(np.sum(np.cumsum(hits) / ranks * hits) / denom),
            'ndcg': float(np.sum(hits / np.log2(ranks + 1)) /
                          np.sum(1 / np.log2(np.arange(1, denom + 1) + 1)))}


def diversity(ids, genres):
    """Mean pairwise Jaccard distance between genre sets."""
    pairs = [1 - len(genres[a] & genres[b]) / max(1, len(genres[a] | genres[b]))
             for j, a in enumerate(ids) for b in ids[j + 1:]]
    return float(np.mean(pairs)) if pairs else 0.0
