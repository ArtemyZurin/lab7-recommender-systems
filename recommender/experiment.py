from pathlib import Path
import importlib.metadata
import json
import platform
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .data import load_data, temporal_split, sha256
from .models import Context, Recommender
from .metrics import ranking_metrics, diversity

MODELS = ['popular', 'content', 'svd', 'als', 'item_knn']


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def segment(n):
    return 'cold' if n == 0 else 'short' if n < 20 else 'active'


def evaluate(model, holdout, movies, ks):
    c = model.c
    relevant = holdout[holdout.rating >= 4].groupby('userId').movieId.apply(set).to_dict()
    genres = {int(r.movieId): set(r.genres.split('|')) - {'(no genres listed)'} for r in movies.itertuples()}
    recs, rows = {}, []
    for user, truth in sorted(relevant.items()):
        ids = model.recommend(int(user), max(ks))
        recs[int(user)] = ids
        history_genres = set().union(*(genres[i] for i in c.liked.get(user, [])))
        for k in ks:
            selected = ids[:k]
            rows.append({'userId': int(user), 'segment': segment(c.counts.get(user, 0)),
                         'k': k, **ranking_metrics(selected, truth, k),
                         'diversity': diversity(selected, genres),
                         'genre_novelty': (float(np.mean([bool(genres[i] - history_genres) for i in selected]))
                                           if history_genres else np.nan)})
    frame = pd.DataFrame(rows)
    aggregates = []
    for k in ks:
        for group in ['all', 'cold', 'short', 'active']:
            subset = frame[(frame.k == k) & ((frame.segment == group) if group != 'all' else True)]
            if subset.empty:
                continue
            ids = set(i for u in subset.userId for i in recs[int(u)][:k])
            row = {'model': model.kind, 'segment': group, 'k': k, 'users': len(subset),
                   **{m: float(subset[m].mean()) for m in ['precision', 'recall', 'map', 'ndcg', 'diversity', 'genre_novelty']},
                   'coverage': len(ids) / len(c.items), 'unique_items': len(ids)}
            aggregates.append(row)
    return pd.DataFrame(aggregates), frame, recs


def plots(ratings, train, movies, metrics, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.1))
    counts = ratings.rating.value_counts().sort_index()
    ax[0].bar(counts.index, counts.values, width=.35, color='#245b83')
    ax[0].set(title='Распределение оценок', xlabel='Оценка', ylabel='Количество')
    ax[1].hist(ratings.groupby('userId').size(), bins=35, color='#318c83')
    ax[1].set(title='Активность пользователей', xlabel='Оценок на пользователя', ylabel='Пользователи')
    ax[2].hist(ratings.groupby('movieId').size(), bins=35, color='#bc7c38', log=True)
    ax[2].set(title='Популярность фильмов', xlabel='Оценок на фильм', ylabel='Фильмы, log')
    fig.tight_layout(); fig.savefig(output / 'eda.png', dpi=180); plt.close(fig)
    users = {u: n for n, u in enumerate(sorted(ratings.userId.unique()))}
    items = {i: n for n, i in enumerate(sorted(movies.movieId))}
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.scatter(ratings.movieId.map(items), ratings.userId.map(users), s=.2, alpha=.3, rasterized=True)
    ax.set(title='Разреженная матрица взаимодействий: полный набор', xlabel='Индекс фильма', ylabel='Индекс пользователя')
    fig.tight_layout(); fig.savefig(output / 'sparsity.png', dpi=180); plt.close(fig)
    overall = metrics[(metrics.segment == 'all') & (metrics.k == 10)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, metric, title in zip(axes, ['ndcg', 'coverage', 'diversity'], ['NDCG@10', 'Покрытие@10', 'Разнообразие@10']):
        ax.bar(overall.model, overall[metric], color=['#8b959e', '#318c83', '#245b83', '#bc7c38', '#73588a'])
        ax.set_title(title); ax.tick_params(axis='x', rotation=25); ax.grid(axis='y', alpha=.2)
    fig.tight_layout(); fig.savefig(output / 'comparison.png', dpi=180); plt.close(fig)


def run(args):
    started = time.perf_counter()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    ratings, movies = load_data(args.data)
    train, valid, test = temporal_split(ratings)
    chosen = MODELS if args.model == 'all' else [args.model]
    tuning, parameters = [], {}
    vc = Context(train, movies)
    for kind in chosen:
        if kind not in {'svd', 'als'}:
            continue
        for factors in [20, 50, 100]:
            print(f'Validation: {kind}, factors={factors}', flush=True)
            model = Recommender(kind, factors, args.seed).fit(vc)
            aggregate, _, _ = evaluate(model, valid, movies, [10])
            # Prefer warm-user validation: cold users use the same fallback for all models.
            warm = aggregate[aggregate.segment == 'active']
            if warm.empty:
                warm = aggregate[aggregate.segment == 'all']
            tuning.append({'model': kind, 'factors': factors, 'ndcg10': float(warm.ndcg.iloc[0]),
                           'selection_segment': str(warm.segment.iloc[0])})
        trials = [t for t in tuning if t['model'] == kind]
        parameters[kind] = max(trials, key=lambda t: (t['ndcg10'], -t['factors']))['factors']
    pd.DataFrame(tuning, columns=['model', 'factors', 'ndcg10', 'selection_segment']).to_csv(output / 'validation.csv', index=False)
    final_train = pd.concat([train, valid], ignore_index=True)
    c = Context(final_train, movies)
    metrics, per_user, recommendations, timings = [], [], {}, {}
    ks = sorted(set(args.k + [5, 10]))
    for kind in chosen:
        print(f'Test: {kind}', flush=True)
        t = time.perf_counter()
        model = Recommender(kind, parameters.get(kind, 50), args.seed).fit(c)
        fit_time = time.perf_counter() - t
        agg, rows, recs = evaluate(model, test, movies, ks)
        metrics.append(agg); rows['model'] = kind; per_user.append(rows)
        recommendations[kind] = recs
        timings[kind] = {'fit_seconds': fit_time, 'evaluate_seconds': time.perf_counter() - t - fit_time}
    metrics = pd.concat(metrics, ignore_index=True)
    metrics.to_csv(output / 'metrics_wide.csv', index=False, encoding='utf-8-sig')
    metrics.melt(id_vars=['model', 'segment', 'k', 'users'],
                 value_vars=['precision', 'recall', 'map', 'ndcg', 'coverage', 'diversity', 'genre_novelty'],
                 var_name='metric', value_name='value').to_csv(output / 'metrics.csv', index=False, encoding='utf-8-sig')
    per_user = pd.concat(per_user, ignore_index=True)
    per_user.to_csv(output / 'per_user_metrics.csv', index=False, encoding='utf-8-sig')
    write_json(output / 'recommendations_all.json', recommendations)
    rng = np.random.default_rng(args.seed)
    intervals = []
    base = per_user[(per_user.model == 'popular') & (per_user.k == 10)]
    if not base.empty:
        for kind in chosen:
            if kind == 'popular': continue
            candidate = per_user[(per_user.model == kind) & (per_user.k == 10)]
            for group in ['all', 'active']:
                joined = base.merge(candidate, on='userId', suffixes=('_base', '_model'))
                if group != 'all': joined = joined[joined.segment_base == group]
                if joined.empty: continue
                diff = (joined.ndcg_model - joined.ndcg_base).to_numpy()
                means = diff[rng.integers(len(diff), size=(2000, len(diff)))].mean(axis=1)
                lo, hi = np.quantile(means, [.025, .975])
                intervals.append({'model': kind, 'segment': group, 'users': len(diff),
                                  'delta_ndcg10': float(diff.mean()), 'ci_low': float(lo), 'ci_high': float(hi)})
    pd.DataFrame(intervals).to_csv(output / 'bootstrap.csv', index=False)
    info = movies.set_index('movieId').to_dict('index')
    users = sorted(next(iter(recommendations.values())))
    active = sorted((u for u in users if c.counts.get(u, 0) >= 20), key=lambda u: (-c.counts[u], u))
    # Two examples fixed by training activity, not test success.
    selected = active[:1] + (active[-1:] if len(active) > 1 else [])
    samples = []
    for u in selected:
        history = final_train[final_train.userId == u].sort_values(['timestamp', 'movieId'])
        truth = set(test[(test.userId == u) & (test.rating >= 4)].movieId)
        samples.append({'userId': u, 'training_ratings': len(history),
                        'test_relevant_count': len(truth),
                        'history': [{'movieId': int(r.movieId), 'rating': float(r.rating), **info[int(r.movieId)]} for r in history.itertuples()],
                        'recommendations': {kind: [{'movieId': i, **info[i], 'hit': i in truth} for i in recommendations[kind][u][:10]] for kind in chosen}})
    write_json(output / 'recs.json', samples)
    splits = ratings[['userId', 'movieId', 'timestamp']].copy()
    splits['split'] = np.where(splits.timestamp < int(valid.timestamp.min()), 'train',
                               np.where(splits.timestamp < int(test.timestamp.min()), 'validation', 'test'))
    splits.to_csv(output / 'split.csv', index=False)
    positives = test[test.rating >= 4]
    audit = {
        'ratings': len(ratings), 'users': int(ratings.userId.nunique()), 'catalog_items': len(movies),
        'rated_items': int(ratings.movieId.nunique()),
        'sparsity': 1 - len(ratings) / (ratings.userId.nunique() * len(movies)),
        'median_user_ratings': float(ratings.groupby('userId').size().median()),
        'median_item_ratings': float(ratings.groupby('movieId').size().median()),
        'positive_ratings': int((ratings.rating >= 4).sum()),
        'partitions': {name: {'ratings': len(df), 'users': int(df.userId.nunique()),
                            'items': int(df.movieId.nunique()), 'start_utc': str(pd.to_datetime(df.timestamp.min(), unit='s', utc=True)),
                            'end_utc': str(pd.to_datetime(df.timestamp.max(), unit='s', utc=True))}
                       for name, df in [('train', train), ('validation', valid), ('test', test)]},
        'final_train_ratings': len(final_train), 'candidate_items': len(c.items),
        'test_users': int(test.userId.nunique()), 'evaluated_users': len(users),
        'excluded_no_test_positive': int(test.userId.nunique()) - len(users),
        'cold_users': sum(u not in c.umap for u in users),
        'unreachable_test_positives': int((~positives.movieId.isin(c.items)).sum()),
        'test_positives': len(positives), 'selected_examples': selected,
        'seed': args.seed, 'parameters': parameters, 'k': ks, 'timings': timings,
        'feedback': 'SVD: explicit 0.5..5; ALS/item_knn: binary rating>=4, not real view logs',
        'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'scikit-surprise', 'implicit', 'matplotlib']},
        'python': platform.python_version(), 'platform': platform.platform(),
        'code_sha256': {str(p.relative_to(Path(__file__).resolve().parents[1])): sha256(p)
                        for p in sorted(Path(__file__).resolve().parent.glob('*.py'))},
        'input_sha256': {'ratings.csv': sha256(Path(args.data) / 'ml-latest-small/ratings.csv'),
                         'movies.csv': sha256(Path(args.data) / 'ml-latest-small/movies.csv')},
        'elapsed_seconds': time.perf_counter() - started}
    write_json(output / 'summary.json', audit)
    plots(ratings, final_train, movies, metrics, output)
    print(metrics[(metrics.segment == 'all') & (metrics.k == 10)].to_string(index=False), flush=True)
