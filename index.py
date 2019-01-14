import sys
from subprocess import call
from clustering import load_model, _create_key_words_for_cluster
# for example python3 index.py some_id

import sqlite3
from db_api import get_field_by_id, add_cluster, create_cluster_info, get_data, get_present_by_id, get_user_by_id
from word2vec_clf import find_most_similar_class

db = sqlite3.connect('db/users.db')
cursor = db.cursor()

allowed_fields = ['about', 'activities', 'books', 'communities',
                  'games', 'interests', 'inspired_by', 'movies',
                  'music', 'status']


def create_cluster_vec(id):
    data = get_user_by_id(cursor, id)
    keys = [key for key in data.keys() if key in allowed_fields]

    for key in keys:
        if key == 'inspired_by':
            field_name = 'personal_inspired_by'
        else:
            field_name = key

        current_model = load_model(field_name + '.pkl')
        current_vectorizer = load_model(field_name + '_vec.pkl')

        if key != 'communities':
            X = current_vectorizer.transform([' '.join(data[key])])
            predicts = current_model.predict(X).tolist()
            prediction = predicts[0]
            print(prediction)
            # save predictions to db
            # add_cluster(cursor, key, prediction, id)
        else:
            for idx, com in enumerate(data['communities']):
                print(idx, com)
                X = current_vectorizer.transform([' '.join(com)])
                predicts = current_model.predict(X).tolist()
                prediction = predicts[0]
                print(prediction)
                # save predictions to db
                # add_cluster(cursor, 'com_' + str(idx), prediction, id)


def get_word2vec_class(id):
    dict = get_user_by_id(cursor, id)
    united_values = [' '.join(dict[key]) for key in dict.keys() if key in allowed_fields]
    flattened = [sublist for sublist in united_values]
    res_str = ' '.join(flattened)

    key, value, words = find_most_similar_class(res_str)
    print(key, value, words)
    return key

id = sys.argv[1]
call('node addPersonById.js ' + str(id), cwd='/home/ftlka/Documents/diploma/fetcher', shell=True)
print('predicting...')

create_cluster_vec(id)

# we need only X for now
# X, y_and_id = get_data(cursor, id)
# print(X)

# bayes_clf = load_model('naive_b.pkl')
# tree_clf = load_model('forest.pkl')
# boost_clf = load_model('boost.pkl')
#
# present_id = bayes_clf.predict(X)[0]
# present_forest_id = tree_clf.predict(X)[0]
# boost_present_id = boost_clf.predict(X)[0]
# print('present id is ' + present_id)
# print('tree id:', present_forest_id)
# print('boost present id:', boost_present_id)
#
# print(get_present_by_id(cursor, present_id))
# print(get_present_by_id(cursor, present_forest_id))
# print(get_present_by_id(cursor, boost_present_id))
