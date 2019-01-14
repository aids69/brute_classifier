import numpy as np
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

import sqlite3
from db_api import get_records_by_field, add_cluster,\
    format_string, get_group_info, save_community, get_communities_info,\
    get_all_communities_info, get_first_communities_info

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
# cluster_amounts = [4, 5, 3, 35, 3, 5, 3, 2, 4, 5]
cluster_fields = ['communities']
cluster_amounts = [40]
# cluster_fields = ['about', 'activities', 'books',
#                   'games', 'interests', 'personal_inspired_by', 'movies',
#                   'music', 'status']
# cluster_amounts = [7, 12, 6, 3, 6, 6, 9, 7, 9]
# cluster_fields = ['about']
# cluster_amounts = [3]


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


def _process_data(data):
    """Removes special characters, empty strings and creates tuple(id, info)"""
    for idx, user in enumerate(data):
        data[idx] = tuple([user[0], format_string(user[1])])

    return [x for x in data if x[1]]


def _process_communities_data(data, segment, seg_start=-1):
    """Fetches data for each community and removes special characters from it"""
    all_users = [x for x in data if x[1]]

    if seg_start == -1:
        seg_start = segment - 0.05

    current_users_seg = all_users[int(seg_start * len(all_users)):int(segment * len(all_users))]
    print('Current segment:', seg_start, segment)
    print('Total:', len(current_users_seg))

    for idx, user in enumerate(current_users_seg):
        if idx % 2500 == 0:
            print(str(100 * idx / len(current_users_seg)) + '%')

        communities = user[1].split(',')[:26]
        communities_info = [None] * len(communities)
        for i, id in enumerate(communities):
            if id:
                communities_info[i] = get_group_info(cursor, id)
        current_users_seg[idx] = tuple([user[0], communities_info])

    return current_users_seg


def _fit_and_save_models(data, cluster_amount, file_name, vec_file_name):
    """Fits vectorizer and kmeans and saves them"""
    ids, words = map(list, zip(*data))

    vectorizer = TfidfVectorizer(max_df=0.1, min_df=0.0005)
    flattened = [' '.join(sublist) for sublist in words]
    X = vectorizer.fit_transform(flattened)

    model = KMeans(n_clusters=cluster_amount)
    model.fit(X)

    save_model(model, file_name)
    save_model(vectorizer, vec_file_name)


def _fit_and_save_com_models(cluster_amount):
    """Fits communities vectorizers and models and saves them"""
    all_info = get_all_communities_info(cursor)[:300000]
    print('got data')
    all_info = list(set([' '.join(el[1:]).strip() for el in all_info]))
    print('Data is ready')
    print(len(all_info))
    vectorizer = TfidfVectorizer(max_df=0.35, min_df=0.025)
    vectorizer.fit(all_info)
    print('Data is fit')
    save_model(vectorizer, 'communities_vec.pkl')
    del all_info, vectorizer
    print('Vectorizer is saved')
    # return

    current_communities = [el for el in get_first_communities_info(cursor)]
    flat_list = [item for sublist in current_communities for item in sublist if item]
    print(len(flat_list))

    vectorizer = load_model('communities_vec.pkl')
    X = vectorizer.transform(flat_list)
    del current_communities, flat_list, vectorizer

    print('Created vectorizer')

    model = KMeans(n_clusters=cluster_amount, verbose=True, init='random')
    model.fit(X)
    save_model(model, 'communities.pkl')
    print('Created kmeans')


def _predict_and_save(data, field, model, vectorizer):
    """Flattens and vectorizes data, makes predictions and saves them"""
    ids, words = map(list, zip(*data))

    flattened = [' '.join(sublist) for sublist in words]
    X = vectorizer.transform(flattened)

    predicts = model.predict(X)
    # save predictions to db
    for i in range(len(ids)):
        add_cluster(cursor, field, predicts[i], ids[i])


def _predict_and_save_communities():
    """Flattens and vectorizes data, makes prediction and saves it"""
    ids = [el[0] for el in get_communities_info(cursor, 'id')]
    # for group intervals used in mark_data.py
    points = range(26)

    for point in points:
        # slicing some range of communities
        current_communities = [el[0] for el in get_communities_info(cursor, 'com_' + str(point))]
        print(point)

        model = load_model('communities.pkl')
        vectorizer = load_model('communities_vec.pkl')

        X = vectorizer.transform(current_communities)
        predicts = model.predict(X)
        print('Applied models, saving to db')

        # save predictions to db
        field = 'communities_' + str(point)
        for i in range(len(ids)):
            add_cluster(cursor, field, predicts[i], ids[i])


def create_and_save_models():
    """Iterates through cluster field and creates kmeans clusters, then saves them to files"""
    for i, cluster_field in enumerate(cluster_fields):
        print(cluster_field)
        current_file_name = cluster_field + '.pkl'
        vec_file_name = cluster_field + '_vec.pkl'

        if cluster_field != 'communities':
            data = get_records_by_field(cursor, cluster_field)
            data = _process_data(data)
            _fit_and_save_models(data, cluster_amounts[i], current_file_name, vec_file_name)
        else:
            _fit_and_save_com_models(cluster_amounts[i])


def apply_saved_models():
    """Loads already saved models, marks data and saves marks and clusters to db"""
    for i, cluster_field in enumerate(cluster_fields):
        print(cluster_field)
        data = get_records_by_field(cursor, cluster_field)

        if cluster_field != 'communities':
            current_model = load_model(cluster_field + '.pkl')
            current_vectorizer = load_model(cluster_field + '_vec.pkl')

            data = _process_data(data)
            _predict_and_save(data, cluster_field, current_model, current_vectorizer)

        else:
            _predict_and_save_communities()


def save_communities():
    """Formats communities and saves them to separate table to save time"""
    data = get_records_by_field(cursor, 'communities')

    for segment in np.arange(0.05, 1.05, 0.05):
        current_seg = _process_communities_data(data, segment)
        ids, words = map(list, zip(*current_seg))
        # for group intervals used in mark_data.py
        points = range(26)

        for idx, id in enumerate(ids):
            current_community = [''] * 26
            for point in points:
                # if there's not enough communities
                if point >= len(words[idx]):
                    break
                flattened = ' '.join(words[idx][point])
                # adding to communities array
                current_community[point] = flattened
            save_community(cursor, id, current_community)


# save_communities()
# create_and_save_models()
# apply_saved_models()


db.commit()
db.close()
