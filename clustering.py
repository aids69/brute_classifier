from sklearn.externals import joblib
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer

import sqlite3
from db_api import get_records_by_field, add_cluster, create_cluster_info, get_user

db = sqlite3.connect('db/users.db')
cursor = db.cursor()


def save_model(model, file_name):
    """Saves model to a specific file"""
    joblib.dump(model, './models/' + file_name)


def load_model(file_name):
    """Loads model from a file it was saved to"""
    model = joblib.load('./models/' + file_name)
    return model


# cluster_fields = ['about', 'activities', 'books', 'communities',
#                   'games', 'interests', 'personal_inspired_by', 'movies',
#                   'music', 'status']
# cluster_amounts = [7, 12, 6, 20, 3, 6, 6, 9, 7, 9]
cluster_fields = ['communities']
cluster_amounts = [20]
# cluster_fields = ['about', 'activities', 'books',
#                   'games', 'interests', 'personal_inspired_by', 'movies',
#                   'music', 'status']
# cluster_amounts = [7, 12, 6, 3, 6, 6, 9, 7, 9]
# cluster_fields = ['about']
# cluster_amounts = [3]


def create_and_save_models():
    """Iterates through cluster field and creates kmeans clusters, then saves them to files"""
    for i, cluster_field in enumerate(cluster_fields):
        print(cluster_field)
        current_file_name = cluster_field + '.pkl'
        vec_file_name = cluster_field + '_vec.pkl'
        data = get_records_by_field(cursor, cluster_field)
        ids, words = map(list, zip(*data))

        vectorizer = TfidfVectorizer(max_df=0.01, min_df=0.005)
        flattened = [' '.join(sublist) for sublist in words]
        X = vectorizer.fit_transform(flattened)

        true_k = cluster_amounts[i]
        model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
        model.fit(X)
        # if cluster_field != 'communities':
        #     data = get_records_by_field(cursor, cluster_field)
        #     ids, words = map(list, zip(*data))
        #
        #     vectorizer = TfidfVectorizer(max_df=0.01, min_df=0.005)
        #     flattened = [' '.join(sublist) for sublist in words]
        #     X = vectorizer.fit_transform(flattened)
        #
        #     true_k = cluster_amounts[i]
        #     model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
        #     model.fit(X)
        #
        # else:
        #     model, vectorizer = get_records_by_field(cursor, cluster_field,
        #                                              MiniBatchKMeans(n_clusters=cluster_amounts[i]),
        #                                              # TfidfVectorizer(max_df=0.75, min_df=0.005))
        #                                              HashingVectorizer())
        save_model(model, current_file_name)
        save_model(vectorizer, vec_file_name)


def _create_key_words_for_cluster(model, terms, clusters_amount):
    """Returns array of clusters each with 30 most important words"""
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    res = []
    for i in range(clusters_amount):
        arr = []
        for ind in order_centroids[i, :30]:
            arr.append(terms[ind])
        res.append(arr)
    return res


def apply_saved_models():
    """Loads already saved models, marks data and saves marks and clusters to db"""
    for i, cluster_field in enumerate(cluster_fields):
        print(cluster_field)
        current_model = load_model(cluster_field + '.pkl')
        current_vectorizer = load_model(cluster_field + '_vec.pkl')

        data = get_records_by_field(cursor, cluster_field)
        ids, words = map(list, zip(*data))
        clusters_amount = cluster_amounts[i]

        flattened = [' '.join(sublist) for sublist in words]
        X = current_vectorizer.transform(flattened)

        predicts = current_model.predict(X)
        # save predictions to db
        for i in range(len(ids)):
            add_cluster(cursor, cluster_field, predicts[i], ids[i])
        # save cluster info to db
        key_words = _create_key_words_for_cluster(current_model,
                                                  current_vectorizer.get_feature_names(),
                                                  clusters_amount)
        create_cluster_info(cursor, cluster_field, key_words)
        # if cluster_field != 'communities':
        #     ids, words = map(list, zip(*data))
        #     clusters_amount = cluster_amounts[i]
        #
        #     flattened = [' '.join(sublist) for sublist in words]
        #     X = current_vectorizer.transform(flattened)
        #
        #     predicts = current_model.predict(X)
        #     # save predictions to db
        #     for i in range(len(ids)):
        #         add_cluster(cursor, cluster_field, predicts[i], ids[i])
        #     # save cluster info to db
        #     key_words = _create_key_words_for_cluster(current_model,
        #                                               current_vectorizer.get_feature_names(),
        #                                               clusters_amount)
        #     create_cluster_info(cursor, cluster_field, key_words)


create_and_save_models()
apply_saved_models()

db.commit()
db.close()
