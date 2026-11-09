"""Independent checks against persisted recommendations, split and metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from recommender.metrics import ranking_metrics


def verify(output):
    root = Path(__file__).parent
    ratings = pd.read_csv(root / 'data/ml-latest-small/ratings.csv')
    split = pd.read_csv(output / 'split.csv')
    merged = ratings.merge(split, on=['userId', 'movieId', 'timestamp'], validate='one_to_one')
    assert len(merged) == len(ratings)
    train = merged[merged.split != 'test']; test = merged[merged.split == 'test']
    assert train.timestamp.max() < test.timestamp.min()
    assert merged[merged.split == 'train'].timestamp.max() < merged[merged.split == 'validation'].timestamp.min()
    available = set(train.movieId)
    seen = train.groupby('userId').movieId.apply(set).to_dict()
    truth = test[test.rating >= 4].groupby('userId').movieId.apply(set).to_dict()
    recs = json.loads((output / 'recommendations_all.json').read_text(encoding='utf-8'))
    per = pd.read_csv(output / 'per_user_metrics.csv')
    wide = pd.read_csv(output / 'metrics_wide.csv')
    for model, users in recs.items():
        assert set(map(int, users)) == set(truth)
        for raw_user, ids in users.items():
            user = int(raw_user)
            assert len(ids) == len(set(ids))
            assert set(ids) <= available
            assert not (set(ids) & seen.get(user, set()))
            for k in per.k.unique():
                row = per[(per.model == model) & (per.userId == user) & (per.k == k)].iloc[0]
                for metric, value in ranking_metrics(ids, truth[user], int(k)).items():
                    assert np.isclose(row[metric], value)
        for row in wide[wide.model == model].itertuples():
            subset = per[(per.model == model) & (per.k == row.k)]
            if row.segment != 'all': subset = subset[subset.segment == row.segment]
            assert len(subset) == row.users
            for metric in ['precision', 'recall', 'map', 'ndcg', 'diversity']:
                assert np.isclose(subset[metric].mean(), getattr(row, metric))
            unique = {i for u in subset.userId for i in users[str(u)][:row.k]}
            assert np.isclose(row.coverage, len(unique) / len(available))
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    assert summary['unreachable_test_positives'] == sum(len(s - available) for s in truth.values())
    message = f'PASS: chronological split, {len(ratings)} ratings, {len(recs)} models, {len(truth)} users; exclusions, ranking metrics, coverage, aggregates and unreachable positives verified.'
    (output / 'verification.txt').write_text(message + '\n', encoding='utf-8')
    print(message)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, default=Path(__file__).parent / 'results')
    verify(p.parse_args().output)
