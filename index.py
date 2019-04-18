import numpy as np
import sqlite3
import sys

from clustering import load_model
from subprocess import call
# for example python3 index.py some_id

from db_api import add_cluster, get_data, get_user_by_id
from mark_data import mark_next_free_person
from word2vec_clf import find_most_similar_class


def create_cluster_vec(cursor, id):
    allowed_fields = ['about', 'activities', 'books', 'communities',
                      'games', 'interests', 'inspired_by', 'movies',
                      'music', 'status']

    imp = load_model('imp.pkl')
    clusters = imp.statistics_

    data = get_user_by_id(cursor, id)
    keys = list(set(data.keys()) & set(allowed_fields))

    if 'communities' in keys:
        keys.remove('communities')
        amount_of_coms = len(data['communities'])
        arr_of_coms = ['com_' + str(e) for e in range(amount_of_coms)]
        allowed_fields += arr_of_coms

    for key in keys:
        if key == 'inspired_by':
            field_name = 'personal_inspired_by'
        else:
            field_name = key

        if key != 'communities':
            current_model = load_model(field_name + '.pkl')
            current_vectorizer = load_model(field_name + '_vec.pkl')

            X = current_vectorizer.transform([' '.join(data[key])])
            predicts = current_model.predict(X).tolist()
            prediction = predicts[0]
            insert_pos = allowed_fields.index(key)
            clusters[insert_pos] = prediction
        else:
            current_model = load_model('mini.pkl')
            current_vectorizer = load_model('communities_vec.pkl')
            pca = load_model('pca.pkl')

            for idx, com in enumerate(data['communities']):
                name = 'com_' + str(idx)
                print(idx, com)
                X = current_vectorizer.transform([' '.join(com)])
                predicts = current_model.predict(pca.transform(X)).tolist()
                prediction = predicts[0]
                insert_pos = allowed_fields.index(name)
                clusters[insert_pos] = prediction
    field_names = ['about', 'activities', 'books',
                   'games', 'interests', 'personal_inspired_by', 'movies',
                   'music', 'status']
    field_names += ['communities_' + str(e) for e in range(26)]
    for i, cluster in enumerate(clusters):
        add_cluster(cursor, field_names[i], cluster, id)


def get_word2vec_class(cursor, id):
    allowed_fields = ['about', 'activities', 'books', 'communities',
                      'games', 'interests', 'inspired_by', 'movies',
                      'music', 'status']

    dict = get_user_by_id(cursor, id)
    united_values = [' '.join(dict[key]) for key in dict.keys() if key in allowed_fields]
    flattened = [sublist for sublist in united_values]
    res_str = ' '.join(flattened)

    key, value, words = find_most_similar_class(res_str)
    print(key, value, words)
    return key


id = sys.argv[1]


def proccess_req(id):
    db = sqlite3.connect('db/users.db', timeout=10)
    cursor = db.cursor()

    call('node addPersonById.js ' + str(id), cwd='/home/ftlka/Documents/diploma/fetcher', shell=True)
    # for cases when screen_name is passed
    with open('current_id.txt', 'r') as f:
        id = str(f.read())
    print('predicting...')

    create_cluster_vec(cursor, id)
    print('created cluster')

    brute = mark_next_free_person(cursor, id)
    print(brute)

    w2v = get_word2vec_class(cursor, id)
    print(w2v)

    # we need only X for now
    X, y_and_id = get_data(cursor, id)
    print(X)
    X = np.delete(X, 3, 1)

    tree_clf = load_model('forest.pkl')
    forest_present_id = tree_clf.predict(X)[0]
    print(forest_present_id)

    with open('../simple_interface/results.txt', 'w') as file:
        file.write(str(brute) + ' ' + str(forest_present_id) + ' ' + str(w2v))

    db.commit()
    db.close()


proccess_req(id)
