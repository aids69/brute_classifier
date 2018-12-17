from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

import sqlite3
from db_api import get_records_by_field

db = sqlite3.connect('db/users.db')
cursor = db.cursor()


def save_model(model, file_name):
    """Saves model to a specific file"""
    joblib.dump(model, file_name)


def load_model(file_name):
    """Loads model from a file it was saved to"""
    model = joblib.load(file_name)
    return model


cluster_fields = ['about', 'activities', 'books', 'communities',
                  'games', 'interests', 'personal_inspired_by', 'movies',
                  'music', 'status']
cluster_amounts = [7, 12, 6, 20, 3, 6, 6, 9, 7, 9]

for i, cluster_field in cluster_fields:
    current_file_name = cluster_field + '.pkl'
    data = get_records_by_field(cursor, cluster_field)
    ids, words = map(list, zip(*data))

    vectorizer = TfidfVectorizer()
    flattened = [' '.join(sublist) for sublist in words]
    X = vectorizer.fit_transform(flattened)

    true_k = cluster_amounts[i]
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model.fit(X)
    save_model(model, current_file_name)
    break


