import re
import numpy as np

pattern = re.compile('([^\s\w]|_)+')


def _format_string(str):
    """Removes special characters and numbers and returns list of words"""
    if isinstance(str, int):
        return str

    new_str = ' '.join(str.split('\\n'))
    new_str = re.sub(r'\d+', '', pattern.sub('', new_str.lower()))
    return list(filter(None, new_str.split()))


def _get_group_info(crs, id):
    """Gets group info by id, returns list of formatted words"""
    all_users = crs.execute('SELECT * FROM groups WHERE id=' + id)
    current_group = all_users.fetchone()
    if current_group == None:
        return ['']

    res = []
    # 2 -> name, 9 -> description, 10 -> status
    indices = [2, 9, 10]
    for i in indices:
        if current_group[i]:
            res += _format_string(current_group[i])

    return res


def get_key_words(crs):
    """Gets key words from presents table"""
    dict = {}
    all_presents = crs.execute('SELECT * FROM presents').fetchall()
    for present in all_presents:
        dict[present[-1]] = present[-2].split()
    return dict


def get_user(crs):
    """
        Gets first user with groups and no present selected yet
        returns informative fields + id
    """
    all_users = crs.execute('SELECT * FROM users')
    current_user = all_users.fetchone()

    # looking for user without present and with groups
    while current_user[117] or current_user[18] == '-' or len(current_user[18].split(',')) < 10:
        current_user = all_users.fetchone()

    params = {
        4: 'about', 5: 'activities', 7: 'books',
        30: 'games', 37: 'interests', 48: 'movies', 49: 'music',
        60: 'inspired_by', 65: 'quotes', 85: 'sex', 88: 'status'
    }
    fields = {'id': current_user[0]}

    communities_info = []
    communities = current_user[18].split(',')
    for id in communities:
        communities_info.append(_get_group_info(crs, id))
    fields['communities'] = communities_info

    for key, value in params.items():
        if current_user[key]:
            fields[value] = _format_string(current_user[key])

    return fields


def assign_present(crs, id, present_id):
    """Adds present for specified user"""
    crs.execute('UPDATE users SET present_id = ' + str(present_id) + ' WHERE id = ' + str(id))


def drop_presents(crs):
    """Resets presents column to empty"""
    crs.execute('UPDATE users SET present_id = NULL')


def add_prediction(crs, id, present_id):
    """Adds brute prediction"""
    crs.execute('INSERT OR IGNORE INTO classes(person_id, brute) VALUES(' +
                str(id) + ', ' + str(present_id) + ')')
    if crs.rowcount == 0:
        crs.execute('UPDATE classes SET brute = ' + str(present_id) + ' WHERE person_id = ' + str(id))


from sklearn.externals import joblib


def save_model(model, file_name):
    """Saves model to a specific file"""
    joblib.dump(model, './models/' + file_name)


def load_model(file_name):
    """Loads model from a file it was saved to"""
    model = joblib.load('./models/' + file_name)
    return model


def _get_all_info_by_field(crs, field_name):
    users = crs.execute('SELECT id, ' + field_name + ' FROM users WHERE ' + field_name + ' IS NOT NULL')
    all_users = users.fetchall()

    for idx, user in enumerate(all_users):
        all_users[idx] = tuple([user[0], _format_string(user[1])])

    # filtering empty strings because they are not NULL
    all_users = [x for x in all_users if x[1]]
    return all_users


def get_records_by_field(crs, field_name, clf=None, vec=None):
    """Gets non-empty data by field from all users or by id"""
    if field_name != 'communities':
        return _get_all_info_by_field(crs, field_name)

    # elif clf is None:
    else:
        users = crs.execute('SELECT id, communities FROM users WHERE communities <> "-"')
        all_users = users.fetchall()
        all_users = [x for x in all_users if x[1]]

        for segment in np.arange(0.05, 1.05, 0.05):
            # current_users_seg = all_users[int((segment - 0.05) * len(all_users)) : int(segment * len(all_users))]
            current_users_seg = all_users[0:int(0.1 * len(all_users))]
            print('total:', len(current_users_seg))
            print(segment - 0.05, segment)
            for idx, user in enumerate(current_users_seg):
                if idx % 1000 == 0:
                    print(str(100 * idx / len(current_users_seg)) + '%')
                communities = user[1].split(',')[:25]
                communities_info = [None] * len(communities)
                for i, id in enumerate(communities):
                    if id:
                        communities_info[i] = _get_group_info(crs, id)
                communities_info = [item for sublist in communities_info for item in sublist]
                current_users_seg[idx] = tuple([user[0], communities_info])
                # filtering empty strings because they are not NULL
            # тут добавила строку
            return [x for x in current_users_seg if x[1]]
            ids, words = map(list, zip(*[x for x in current_users_seg if x[1]]))

            flattened = [' '.join(sublist) for sublist in words]
            vec = load_model('communities_vec.pkl')
            clf = load_model('communities.pkl')
            X = vec.transform(flattened)

            predicts = clf.predict(X)
            # save predictions to db
            for i in range(len(ids)):
                add_cluster(crs, 'communities', predicts[i], ids[i])
            del ids, words, X, predicts, flattened, current_users_seg, vec, clf
    # else:
    #     users = crs.execute('SELECT id, communities FROM users WHERE communities <> "-"')
    #     all_users = users.fetchall()
    #     all_users = [x for x in all_users if x[1]]
    #
    #     save_model(clf, 'communities.pkl')
    #     del clf
    #
    #     for segment in np.arange(0.05, 1.05, 0.05):
    #         current_users_seg = all_users[int((segment - 0.05) * len(all_users)) : int(segment * len(all_users))]
    #         print(segment - 0.05, segment)
    #         print(int((segment - 0.05) * len(all_users)), int(segment * len(all_users)))
    #         print('total:', len(current_users_seg))
    #         for idx, user in enumerate(current_users_seg):
    #             if idx % 1000 == 0:
    #                 print(str(100 * idx / len(current_users_seg)) + '%')
    #             communities = user[1].split(',')[:25]
    #             communities_info = [None] * len(communities)
    #             for i, id in enumerate(communities):
    #                 if id:
    #                     communities_info[i] = _get_group_info(crs, id)
    #             communities_info = [item for sublist in communities_info for item in sublist]
    #             current_users_seg[idx] = tuple([user[0], communities_info])
    #         # filtering empty strings because they are not NULL
    #         current_users_seg = [x for x in current_users_seg if x[1]]
    #         ids, words = map(list, zip(*current_users_seg))
    #         flattened = [' '.join(sublist) for sublist in words]
    #
    #         clf = load_model('communities.pkl')
    #         clf = clf.partial_fit(vec.transform(flattened))
    #         save_model(clf, 'communities.pkl')
    #         if segment != 1:
    #             del clf, flattened, words, ids, current_users_seg
    #     return clf, vec


def get_field_by_id(crs, field_name, person_id):
    """Returns data for specific field for a specific person"""
    if field_name != 'communities':
        user = crs.execute('SELECT ' + field_name + ' FROM users WHERE id = ' + person_id)
        current_user = user.fetchone()
        if current_user[0] == None:
            return []

        current_user = _format_string(current_user[0])
    else:
        user = crs.execute('SELECT communities FROM users WHERE id = ' + person_id)
        current_user = user.fetchone()
        # current_user = [x for x in current_user if x[1]]
        communities = current_user[0].split(',')
        communities_info = [None] * len(communities)
        for i, id in enumerate(communities):
            communities_info[i] = _get_group_info(crs, id)
        communities_info = [item for sublist in communities_info for item in sublist]
        current_user = communities_info

    return current_user


def _create_sql_values(ids, values):
    """Creates concatenated string of values so there's no need to call INSERT in loop"""
    arr = []
    arr.extend(['(' + str(ids[i]) + ',' + str(values[i]) + ')' for i in range(len(ids))])
    return ','.join(arr)


def add_cluster(crs, cluster_name, value, id):
    """Adds predicted cluster to classes table for specific id"""
    cluster_name += '_cluster'
    # crs.execute('INSERT OR IGNORE INTO classes(person_id, ' + cluster_name
    #             + ') VALUES ' + '(' + str(id) + ',' + str(value) + ')')
    # if id's already in the table, we need to update
    # if crs.rowcount == 0:
    crs.execute('UPDATE classes SET ' + cluster_name +
                ' = ' + str(value) + ' WHERE person_id = ' + str(id))


def create_cluster_info(crs, cluster_name, key_words_arr):
    """Adds cluster to the cluster table and sets key words per each cluster"""
    clusters_amount = len(key_words_arr)
    res_str = ''
    for i in range(clusters_amount):
        res_str += str(i) + ': ' + ' '.join(key_words_arr[i])
        if i != clusters_amount - 1:
            res_str += ', '

    crs.execute('INSERT OR IGNORE INTO clusters(name, amount_of_clusters, cluster_values) VALUES("' +
                str(cluster_name) + '", ' + str(clusters_amount) + ', "' + res_str + '")')


def get_data(crs, person_id=-1):
    """Gets X and y for our data, where X - clusters and y - brute"""
    # TODO: instead of using hardcoded array of cluster_amounts should get it from db
    if person_id == -1:
        classes = crs.execute('SELECT * FROM classes WHERE brute IS NOT NULL').fetchall()
    else:
        classes = crs.execute('SELECT * FROM classes WHERE person_id = ' + person_id).fetchall()
    # person_id, brute, cluster0, cluster1,... => +2
    cluster_amounts = [7, 12, 6, 20, 3, 6, 6, 9, 7, 9]
    X = []
    y_and_ids = []

    for current_user in classes:
        current_user = np.array(current_user)
        for i, e in enumerate(current_user):
            # if not available we mark it as a max+1 class
            if e == None:
                current_user[i] = cluster_amounts[i-2]
        X.append(np.array(current_user[2:]))
        y_and_ids.append(np.array(current_user[:2]))
    return np.array(X), np.array(y_and_ids)


def get_present_by_id(crs, present_id):
    present = crs.execute('SELECT name FROM presents WHERE id = ' + present_id).fetchone()[0]
    return present
