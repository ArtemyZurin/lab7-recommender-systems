import math
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import ndcg_score
from recommender.metrics import ranking_metrics, diversity
from recommender.data import temporal_split
from recommender.models import Context, Recommender


def test_metrics_known_answer():
    m = ranking_metrics([1, 8, 2, 9], {1, 2, 3}, 4)
    assert m['precision'] == .5
    assert m['recall'] == pytest.approx(2 / 3)
    assert m['map'] == pytest.approx((1 + 2 / 3) / 3)
    assert m['ndcg'] == pytest.approx(1.5 / (1 + 1 / math.log2(3) + .5))


def test_ndcg_matches_sklearn():
    truth = np.array([[1, 0, 1, 0, 1]])
    scores = np.array([[5, 4, 3, 2, 1]])
    assert ranking_metrics([1, 2, 3, 4, 5], {1, 3, 5}, 4)['ndcg'] == pytest.approx(ndcg_score(truth, scores, k=4))


def test_short_list_and_empty_relevance():
    assert ranking_metrics([1], {1, 2}, 5)['precision'] == .2
    with pytest.raises(ValueError): ranking_metrics([1], set(), 5)
    with pytest.raises(ValueError): ranking_metrics([1, 1], {1}, 5)


def test_temporal_ties_do_not_cross_partitions():
    frame = pd.DataFrame({'timestamp': np.repeat(np.arange(20), 2), 'userId': 1,
                          'movieId': np.arange(40), 'rating': 4})
    a, b, c = temporal_split(frame.sample(frac=1, random_state=9))
    assert a.timestamp.max() < b.timestamp.min() <= b.timestamp.max() < c.timestamp.min()
    assert len(a) + len(b) + len(c) == len(frame)


@pytest.fixture
def context():
    frame = pd.DataFrame([(1, 1, 5), (1, 2, 3), (2, 2, 4), (2, 3, 5), (3, 1, 4), (3, 4, 5)],
                         columns=['userId', 'movieId', 'rating'])
    movies = pd.DataFrame({'movieId': [1, 2, 3, 4, 99], 'title': ['One film', 'Two film', 'Three film', 'Four film', 'Future film'],
                           'genres': ['Drama', 'Drama', 'Comedy', 'Comedy', 'Drama']})
    return Context(frame, movies)


@pytest.mark.parametrize('kind', ['popular', 'content', 'svd', 'als', 'item_knn'])
def test_exclude_seen_and_future_and_cold_fallback(context, kind):
    model = Recommender(kind, factors=3).fit(context)
    ids = model.recommend(1, 10)
    assert set(ids) == {3, 4}
    assert len(ids) == len(set(ids))
    assert 99 not in model.recommend(777, 10)
    assert model.recommend(777, 10) == Recommender('popular').fit(context).recommend(777, 10)


def test_svd_vectorization_agrees_with_library(context):
    model = Recommender('svd', factors=3).fit(context)
    expected = [model.model.predict(1, int(i), clip=False).est for i in context.items]
    np.testing.assert_allclose(model.scores(1), expected, atol=1e-12)


def test_diversity():
    assert diversity([1, 2], {1: {'Drama'}, 2: {'Comedy'}}) == 1
    assert diversity([1, 2], {1: {'Drama'}, 2: {'Drama'}}) == 0
