import re

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
    # crs.execute('UPDATE classes SET brute = ' + str(present_id) + ' WHERE person_id = ' + str(id))


def get_records_by_field(crs, field_name):
    """Gets non-empty data by field from all users"""
    if field_name != 'communities':
        users = crs.execute('SELECT id, ' + field_name + ' FROM users WHERE ' + field_name + ' IS NOT NULL')
        all_users = users.fetchall()

        for idx, user in enumerate(all_users):
            all_users[idx] = tuple([user[0], _format_string(user[1])])
    else:
        users = crs.execute('SELECT id, communities FROM users WHERE communities <> "-"')
        all_users = users.fetchall()
        all_users = [x for x in all_users if x[1]]
        # slicing the list down to 5%
        all_users = all_users[: int(len(all_users) * .05)]
        print('total:', len(all_users))

        for idx, user in enumerate(all_users):
            if idx % 5000 == 0:
                print(str(100*idx/len(all_users)) + '%')
            # getting only first 60 communities because my pc dies
            communities = user[1].split(',')[:60]
            communities_info = [None] * len(communities)
            for i, id in enumerate(communities):
                if id:
                    communities_info[i] = _get_group_info(crs, id)
            communities_info = [item for sublist in communities_info for item in sublist]
            all_users[idx] = tuple([user[0], communities_info])

    # filtering empty strings because they are not NULL
    all_users = [x for x in all_users if x[1]]
    return all_users


def _create_sql_values(ids, values):
    """Creates concatenated string of values so there's no need to call INSERT in loop"""
    arr = []
    arr.extend(['(' + str(ids[i]) + ',' + str(values[i]) + ')' for i in range(len(ids))])
    return ','.join(arr)


def add_cluster(crs, cluster_name, value, id):
    """Adds predicted cluster to classes table for specific id"""
    cluster_name += '_cluster'
    crs.execute('INSERT OR IGNORE INTO classes(person_id, ' + cluster_name
                + ') VALUES ' + '(' + str(id) + ',' + str(value) + ')')
    # if id's already in the table, we need to update
    if crs.rowcount == 0:
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

