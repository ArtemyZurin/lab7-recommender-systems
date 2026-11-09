import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from surprise import Dataset, Reader, SVD
from implicit.cpu.als import AlternatingLeastSquares


class Context:
    def __init__(self, train, movies):
        self.train = train
        self.items = np.sort(train.movieId.unique())
        self.users = np.sort(train.userId.unique())
        self.imap = {int(i): j for j, i in enumerate(self.items)}
        self.umap = {int(u): j for j, u in enumerate(self.users)}
        self.seen = train.groupby('userId').movieId.apply(set).to_dict()
        self.counts = train.groupby('userId').size().to_dict()
        pos = train[train.rating >= 4]
        self.liked = pos.groupby('userId').movieId.apply(list).to_dict()
        self.pop = pos.groupby('movieId').size().reindex(self.items, fill_value=0).to_numpy(dtype=float)
        self.binary = csr_matrix((np.ones(len(pos), dtype=np.float32),
                                  (pos.userId.map(self.umap), pos.movieId.map(self.imap))),
                                 shape=(len(self.users), len(self.items)))
        catalog = movies.set_index('movieId').loc[self.items]
        # Only static title/genre metadata: no future user tags or test ratings.
        docs = (catalog.title.str.replace(r'\(\d{4}\)', '', regex=True) + ' ' +
                catalog.genres.str.replace('|', ' ', regex=False))
        self.vectorizer = TfidfVectorizer(min_df=2, max_features=10000, strip_accents='unicode')
        self.features = self.vectorizer.fit_transform(docs)

    def rank(self, scores, user, k):
        scores = np.asarray(scores, dtype=float).copy()
        for item in self.seen.get(user, set()):
            scores[self.imap[item]] = -np.inf
        # Deterministic ties: score, then training popularity, then movieId.
        order = np.lexsort((self.items, -self.pop, -scores))
        return self.items[order[np.isfinite(scores[order])][:k]].astype(int).tolist()


class Recommender:
    def __init__(self, kind, factors=50, seed=42):
        self.kind, self.factors, self.seed = kind, factors, seed

    def fit(self, context):
        self.c = context
        if self.kind == 'svd':
            ds = Dataset.load_from_df(context.train[['userId', 'movieId', 'rating']],
                                      Reader(rating_scale=(.5, 5)))
            self.model = SVD(n_factors=self.factors, n_epochs=20, random_state=self.seed).fit(ds.build_full_trainset())
            self.inner_items = np.array([self.model.trainset.to_inner_iid(int(i)) for i in context.items])
        elif self.kind == 'als':
            self.model = AlternatingLeastSquares(factors=self.factors, regularization=.1,
                                                iterations=20, random_state=self.seed,
                                                num_threads=1, alpha=1)
            # Observed positive confidence c=1+20=21; missing c=1, p=0.
            self.model.fit(context.binary * 21, show_progress=False)
        elif self.kind == 'item_knn':
            items_users = normalize(context.binary.T, norm='l2', axis=1)
            similarity = (items_users @ items_users.T).tocsr()
            similarity.setdiag(0)
            similarity.eliminate_zeros()
            # Keep top 50 neighbours per source item, with stable ties.
            rows, cols, vals = [], [], []
            for i in range(similarity.shape[0]):
                start, end = similarity.indptr[i:i + 2]
                ids, values = similarity.indices[start:end], similarity.data[start:end]
                ix = np.lexsort((ids, -values))[:50]
                rows.extend([i] * len(ix)); cols.extend(ids[ix]); vals.extend(values[ix])
            self.similarity = csr_matrix((vals, (rows, cols)), shape=similarity.shape)
        elif self.kind not in {'popular', 'content'}:
            raise ValueError(self.kind)
        return self

    def scores(self, user):
        c = self.c
        if self.kind == 'popular' or user not in c.umap:
            return c.pop
        liked = c.liked.get(user, [])
        if self.kind in {'content', 'als', 'item_knn'} and not liked:
            return c.pop
        if self.kind == 'content':
            ids = [c.imap[i] for i in liked]
            profile = normalize(np.asarray(c.features[ids].mean(axis=0)))
            return np.asarray(c.features @ profile.T).ravel()
        if self.kind == 'svd':
            model = self.model
            uid = model.trainset.to_inner_uid(user)
            # Same un-clipped estimate as Surprise, vectorized over the catalog.
            return (model.trainset.global_mean + model.bu[uid] + model.bi[self.inner_items]
                    + model.qi[self.inner_items] @ model.pu[uid])
        if self.kind == 'als':
            return self.model.item_factors @ self.model.user_factors[c.umap[user]]
        return (c.binary[c.umap[user]] @ self.similarity).toarray().ravel()

    def recommend(self, user, k=10):
        return self.c.rank(self.scores(user), user, k)
