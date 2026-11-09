from pathlib import Path
import hashlib
import json
import urllib.request
from zipfile import ZipFile

import pandas as pd

URL = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    folder = root / 'ml-latest-small'
    if not (folder / 'ratings.csv').exists():
        archive = root / 'ml-latest-small.zip'
        if not archive.exists():
            urllib.request.urlretrieve(URL, archive)
        with ZipFile(archive) as z:
            # Extract only the five named dataset files, never arbitrary paths.
            folder.mkdir(exist_ok=True)
            for name in ['ratings.csv', 'movies.csv', 'tags.csv', 'links.csv', 'README.txt']:
                (folder / name).write_bytes(z.read('ml-latest-small/' + name))
    ratings = pd.read_csv(folder / 'ratings.csv')
    movies = pd.read_csv(folder / 'movies.csv').sort_values('movieId').reset_index(drop=True)
    if ratings.duplicated(['userId', 'movieId']).any():
        raise ValueError('Duplicate user-item pairs: define an aggregation policy first.')
    if ratings.isna().any().any() or not ratings.rating.between(.5, 5).all():
        raise ValueError('Invalid ratings')
    if not ratings.movieId.isin(movies.movieId).all():
        raise ValueError('Missing item metadata')
    manifest = {'url': URL, 'download_date': '2026-11-09',
                'files': {p.name: sha256(p) for p in sorted(folder.iterdir()) if p.is_file()}}
    if (root / 'ml-latest-small.zip').exists():
        manifest['archive_sha256'] = sha256(root / 'ml-latest-small.zip')
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return ratings.sort_values(['timestamp', 'userId', 'movieId']).reset_index(drop=True), movies


def temporal_split(ratings):
    """Global 80/10/10 time split; equal timestamps stay in one partition."""
    ordered = ratings.sort_values(['timestamp', 'userId', 'movieId'])
    t1 = int(ordered.iloc[int(len(ordered) * .8)].timestamp)
    t2 = int(ordered.iloc[int(len(ordered) * .9)].timestamp)
    train = ordered[ordered.timestamp < t1].copy()
    valid = ordered[(ordered.timestamp >= t1) & (ordered.timestamp < t2)].copy()
    test = ordered[ordered.timestamp >= t2].copy()
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError('Not enough distinct timestamps for temporal evaluation')
    return train, valid, test
