import numpy as np
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import sqlite3
from db_api import get_data
from clustering import save_model
# TODO: возможно проблема в том, что сравниваются строки и числа

db = sqlite3.connect('db/users.db')
cursor = db.cursor()

X, y_and_ids = get_data(cursor)
ids, y = zip(*y_and_ids)
X_train, X_test, y_and_ids_train, y_and_ids_test = train_test_split(X, y_and_ids, test_size=0.3)
ids_train, y_train = zip(*y_and_ids_train)
ids_test, y_test = zip(*y_and_ids_test)

y_train = list(map(int, y_train))
y_test = list(map(int, y_test))
X_train = np.array([[int(j) for j in i] for i in X_train])
X_test = np.array([[int(j) for j in i] for i in X_test])

y_train = np.array(y_train)
y_test = np.array(y_test)
ids_train = np.array(ids_train)
ids_test = np.array(ids_test)

print('y_train', y_train, y_train.shape)
print('y_test', y_test, y_test.shape)
print('X_train', X_train, X_train.shape)
print('X_test', X_test, X_test.shape)

clf = ComplementNB()
clf.fit(X_train, y_train)
save_model(clf, 'naive_b.pkl')
print(accuracy_score(y_test, clf.predict(X_test)))
clf.fit(X, y)
save_model(clf, 'naive_b.pkl')

from sklearn.ensemble import RandomForestClassifier as forest
forest_clf = forest()
forest_clf.fit(X_train, y_train)
print(accuracy_score(y_test, forest_clf.predict(X_test)))
forest_clf.fit(X, y)
save_model(forest_clf, 'forest.pkl')

from sklearn.ensemble import GradientBoostingClassifier
# boost_clf = GradientBoostingClassifier(n_estimators=50)
boost_clf = GradientBoostingClassifier()
boost_clf.fit(X_train, y_train)
print('Gradient boosting', accuracy_score(y_test, boost_clf.predict(X_test)))
boost_clf.fit(X, y)
save_model(boost_clf, 'boost.pkl')

# from sklearn.ensemble import AdaBoostClassifier
# ada_boost_clf = AdaBoostClassifier(boost_clf)
# ada_boost_clf.fit(X_train, y_train)
# print(accuracy_score(y_test, ada_boost_clf.predict(X_test)))


db.commit()
db.close()
