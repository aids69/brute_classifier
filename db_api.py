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
