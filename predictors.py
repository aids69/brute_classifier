import numpy as np
import random
import re
from .keywords import keywords
from .clustering import load_model


pattern = re.compile('([^\s\w]|_)+')


def _preprocess_string(s):
    new_str = ' '.join(s.split('\\n'))
    return re.sub(r'\d+', '', pattern.sub(' ', new_str.lower()))


def _create_com_info_string(com):
    com_info = []
    com_fields = ['name', 'decsription', 'status']

    for com_field in com_fields:
        if com_field in com:
            com_info.append(com[com_field])
    com_info = ' '.join(com_info).lower()

    return _preprocess_string(com_info)


def _count_word(user, communities, key_words):
    total = 0

    # 1 - female, 2 - male
    if 'sex' in user:
        if user['sex'] == 1 and 'мужчин' in key_words:
            return 0
        elif user['sex'] == 2 and\
                len({'макияж', 'девочк', 'мам'}.intersection(key_words)):
            return 0

    for idx, comm in enumerate(communities):
        com_info = _create_com_info_string(comm).split()

        if any([s for s in com_info if any(xs in s for xs in key_words)]):
            if idx == 0:
                total += 30
            elif idx < 2:
                total += 28
            elif idx < 5:
                total += 5
            elif idx < 8:
                total += 1
            elif idx < 26:
                total += 0.1
            else:
                break

    informative_fields = ['about', 'activities', 'interests', 'inspired_by', 'status']
    for field in informative_fields:
        if field in user:
            total += 3 * sum(any(xs in s for xs in key_words) for s in user[field])

    # authors or movie names do not correlate with our key words so we add plus one for being filled
    half_informative_fields = {
        'books': 'книги', 'games': 'игры', 'movies': 'фильмы',
        'music': 'музыка', 'quotes': 'успех'
    }
    half_inf = []
    for key, val in half_informative_fields.items():
        if key in user and user[key]:
            half_inf.append(half_informative_fields[key])
    total += sum(any(xs in s for xs in key_words) for s in half_inf)

    return total


def _create_cluster_vec(user, communities):
    allowed_fields = ['about', 'activities', 'books',
                      'games', 'interests', 'inspired_by', 'movies',
                      'music', 'status']

    # default values
    imp = load_model('imp.pkl')
    clusters = imp.statistics_

    # only non-empty fields
    keys = list({field for field in user if user[field]} & set(allowed_fields))
    if communities:
        keys += ['communities']
        arr_of_coms = ['com_{}'.format(str(e)) for e in range(len(communities))]
        allowed_fields += arr_of_coms

    for key in keys:
        field_name = 'personal_inspired_by' if key == 'inspired_by' else key

        if key != 'communities':
            current_model = load_model(field_name + '.pkl')
            current_vectorizer = load_model(field_name + '_vec.pkl')

            X = current_vectorizer.transform([_preprocess_string(user[key])])
            prediction = current_model.predict(X).tolist()[0]
            insert_pos = allowed_fields.index(key)
            clusters[insert_pos] = str(prediction)
        else:
            current_model = load_model('mini.pkl')
            current_vectorizer = load_model('communities_vec.pkl')
            pca = load_model('pca.pkl')

            for idx, com in enumerate(communities):
                com_info = _create_com_info_string(com)
                X = current_vectorizer.transform([com_info])

                prediction = current_model.predict(pca.transform(X)).tolist()[0]
                insert_pos = allowed_fields.index('com_{}'.format(str(idx)))
                clusters[insert_pos] = str(prediction)
    return clusters


def brute(user, communities):
    max_freq, most_freq_present_id = 0, -1

    for key, value in keywords.items():
        cur_freq = _count_word(user, communities, value)
        if cur_freq >= max_freq:
            max_freq, most_freq_present_id = cur_freq, key

    return 20 if max == 0 else most_freq_present_id


def classifier(user, communities):
    clusters = np.array(_create_cluster_vec(user, communities)).reshape(1, -1)

    tree_clf = load_model('forest.pkl')
    forest_present_id = tree_clf.predict(clusters)[0]
    return forest_present_id


def w2v(user, communities):
    return 3
