import sys
from subprocess import call
from clustering import load_model, _create_key_words_for_cluster
# for example python3 index.py 45047942

import sqlite3
from db_api import get_field_by_id, add_cluster, create_cluster_info, get_data, get_present_by_id

db = sqlite3.connect('db/users.db')
cursor = db.cursor()


def create_cluster_vec(id):
    cluster_fields = ['about', 'activities', 'books', 'communities',
                      'games', 'interests', 'personal_inspired_by', 'movies',
                      'music', 'status']
    cluster_amounts = [7, 12, 6, 20, 3, 6, 6, 9, 7, 9]
    for i, cluster_field in enumerate(cluster_fields):
        current_model = load_model(cluster_field + '.pkl')
        current_vectorizer = load_model(cluster_field + '_vec.pkl')

        data = get_field_by_id(cursor, cluster_field, id)
        words = data

        # flattened = [' '.join(sublist) for sublist in words]
        if len(words) == 0:
            prediction = cluster_amounts[i]
        else:
            X = current_vectorizer.transform(words)
            predicts = current_model.predict(X).tolist()
            prediction = max(set(predicts), key=predicts.count)
        # save predictions to db
        add_cluster(cursor, cluster_field, prediction, id)


id = sys.argv[1]
call("node addPersonById.js " + str(id), cwd="/home/ftlka/Documents/diploma/fetcher", shell=True)
print('predicting...')
create_cluster_vec(id)
bayes_clf = load_model('naive_b.pkl')
tree_clf = load_model('forest.pkl')
boost_clf = load_model('boost.pkl')
# we need only X for now
X, y_and_id = get_data(cursor, id)
print(X)
present_id = bayes_clf.predict(X)[0]
present_forest_id = tree_clf.predict(X)[0]
boost_present_id = boost_clf.predict(X)[0]
print('present id is ' + present_id)
print('tree id:', present_forest_id)
print('boost present id:', boost_present_id)

print(get_present_by_id(cursor, present_id))
print(get_present_by_id(cursor, present_forest_id))
print(get_present_by_id(cursor, boost_present_id))
