import sys
import numpy as np
from subprocess import call
from clustering import load_model
# for example python3 index.py some_id

import sqlite3
from db_api import add_cluster, get_data, get_user_by_id
from word2vec_clf import find_most_similar_class

from mark_data import mark_next_free_person


def create_cluster_vec(cursor, id):
    allowed_fields = ['about', 'activities', 'books', 'communities',
                      'games', 'interests', 'inspired_by', 'movies',
                      'music', 'status']

    imp = load_model('imp.pkl')
    clusters = imp.statistics_

    data = get_user_by_id(cursor, id)
    keys = [key for key in data.keys() if key in allowed_fields]

    if 'communities' in keys:
        keys.remove('communities')
        amount_of_coms = len(data['communities'])
        arr_of_coms = range(amount_of_coms)
        arr_of_coms = ['com_' + str(e) for e in arr_of_coms]
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
    call('node addPersonById.js ' + str(id), cwd='/home/ftlka/Documents/diploma/fetcher', shell=True)
    # for cases when screen_name is passed
    id = str(open('current_id.txt', 'r').read())
    print('predicting...')
    db.commit()

    cursor = db.cursor()
    create_cluster_vec(cursor, id)
    print('created cluster')
    db.commit()
    db.close()
    db = sqlite3.connect('db/users.db', timeout=10)
    cursor = db.cursor()
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

    file = open('../simple_interface/results.txt', 'w')
    file.write(str(brute) + ' ' + str(forest_present_id) + ' ' + str(w2v))

    db.commit()
    db.close()
# print('present id is ' + present_id)
# print('tree id:', present_forest_id)
# print('boost present id:', boost_present_id)
#
# print(get_present_by_id(cursor, present_id))
# print(get_present_by_id(cursor, present_forest_id))
# print(get_present_by_id(cursor, boost_present_id))

proccess_req(id)

# db.commit()
# db.close()
