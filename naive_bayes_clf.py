import numpy as np
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import sqlite3
from db_api import get_data
from clustering import save_model, load_model

db = sqlite3.connect('db/users.db')
cursor = db.cursor()

X, y_and_ids = get_data(cursor)
X_train, X_test, y_and_ids_train, y_and_ids_test = train_test_split(X, y_and_ids, test_size=0.3)
y_train, ids_train = zip(*y_and_ids_train)
y_test, ids_test = zip(*y_and_ids_test)

y_train = np.array(y_train)
y_test = np.array(y_test)
ids_train = np.array(ids_train)
ids_test = np.array(ids_test)

clf = ComplementNB()
clf.fit(X_train, y_train)
save_model(clf, 'naive_b.pkl')
print(accuracy_score(y_test, clf.predict(y_train)))


db.commit()
db.close()
