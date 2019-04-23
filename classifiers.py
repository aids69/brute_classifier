import numpy as np
import sqlite3

from sklearn.ensemble import RandomForestClassifier as forest
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from clustering import save_model
from db_api import get_data


def prepare_data(cursor, test_size=0.3):
    """Returns X_train, X_test, y_train, y_test ready for classification"""
    X, y_and_ids = get_data(cursor)

    # deleting communities - 4th column
    X = np.delete(X, 3, 1)

    imp = SimpleImputer(missing_values=None, strategy='most_frequent', copy=False)
    imp.fit_transform(X)
    save_model(imp, 'imp.pkl')

    X = np.array([[int(x) for x in lst] for lst in X])

    ids, y = zip(*y_and_ids)
    X_train, X_test, y_and_ids_train, y_and_ids_test = train_test_split(X, y_and_ids, test_size=test_size)
    ids_train, y_train = zip(*y_and_ids_train)
    ids_test, y_test = zip(*y_and_ids_test)

    y_train = list(map(int, y_train))
    y_test = list(map(int, y_test))
    X_train = np.array([[int(j) for j in i] for i in X_train])
    X_test = np.array([[int(j) for j in i] for i in X_test])

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print(np.array(y).shape)
    return X_train, X_test, y_train, y_test, X, y, ids_test


def naive_bayes(X_train, X_test, y_train, y_test):
    clf = ComplementNB()
    clf.fit(X_train, y_train)
    print('Naive bayes', accuracy_score(y_test, clf.predict(X_test)))
    # clf.fit(X, y)
    save_model(clf, 'naive_b.pkl')


def mult_bayes(X_train, X_test, y_train, y_test):
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    print('Mult bayes', accuracy_score(y_test, clf.predict(X_test)))
    # clf.fit(X, y)
    save_model(clf, 'naive_b_mult.pkl')


def rand_forest(X_train, X_test, y_train, y_test, ids_test):
    forest_clf = forest(n_estimators=50)
    forest_clf.fit(X_train, y_train)
    pred = forest_clf.predict(X_test)
    print('Random forest', accuracy_score(y_test, pred))
    # forest_clf.fit(X, y)
    save_model(forest_clf, 'forest.pkl')


def grad_boost(X_train, X_test, y_train, y_test):
    boost_clf = GradientBoostingClassifier(n_estimators=50)
    boost_clf.fit(X_train, y_train)
    print('Gradient boosting', accuracy_score(y_test, boost_clf.predict(X_test)))
    # boost_clf.fit(X, y)
    save_model(boost_clf, 'boost.pkl')


def ada_boost(X_train, X_test, y_train, y_test):
    ada_boost_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=100))
    ada_boost_clf.fit(X_train, y_train)
    print('Ada boost', accuracy_score(y_test, ada_boost_clf.predict(X_test)))
    # ada_boost_clf.fit(X, y)
    save_model(ada_boost_clf, 'ada.pkl')


if __name__ == '__main__':
    with sqlite3.connect('db/users.db') as db:
        cursor = db.cursor()
        X_train, X_test, y_train, y_test, X, y, ids_test = prepare_data(cursor)
        naive_bayes(X_train, X_test, y_train, y_test)
        mult_bayes(X_train, X_test, y_train, y_test)
        rand_forest(X_train, X_test, y_train, y_test, ids_test)
        grad_boost(X_train, X_test, y_train, y_test)
        ada_boost(X_train, X_test, y_train, y_test)

        db.commit()
