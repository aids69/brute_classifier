import sqlite3
import random
from db_api import get_key_words, get_user, assign_present, drop_presents, add_prediction

db = sqlite3.connect('db/users.db')
cursor = db.cursor()

key_words = get_key_words(cursor)


def count_word(user, words):
    """Counts substrings of words in user profile and can get maximum +1 count for each community"""
    counter = 0

    # 1 - female, 2 - male
    if 'sex' in user.keys() and user['sex'] == 1 and 'мужчин' in words:
        return 0

    if 'sex' in user.keys() and user['sex'] == 2 and 'макияж' in words:
        return 0

    if 'sex' in user.keys() and user['sex'] == 2 and 'девочк' in words:
        return 0

    if 'sex' in user.keys() and user['sex'] == 2 and 'мам' in words:
        return 0

    idx = 0
    for comm in user['communities']:
        if any([s for s in comm if any(xs in s for xs in words)]):
            if idx == 0:
                counter += 30
            elif idx < 2:
                counter += 28
            elif idx <= 4:
                counter += 5
            elif idx <= 7:
                counter += 1
            elif idx <= 25:
                counter += 0.1
            else:
                counter += 0
        idx += 1

    informative_fields = ['about', 'activities', 'interests', 'inspired_by', 'status']
    for field in informative_fields:
        if field in user.keys():
            counter += 3 * sum(any(xs in s for xs in words) for s in user[field])

    # authors or movie names do not correlate with our key words so we add plus one for being filled
    half_informative_fields = {
        'books': 'книги', 'games': 'игры', 'movies': 'фильмы',
        'music': 'музыка', 'quotes': 'успех'
    }
    half_inf = []
    for key, val in half_informative_fields.items():
        if key in user.keys():
            half_inf.append(half_informative_fields[key])
    counter += sum(any(xs in s for xs in words) for s in half_inf)

    return counter


def mark_next_free_person():
    # make key words' order random for cases with more than 1 max
    keys = list(key_words.keys())
    random.shuffle(keys)

    current_user = get_user(cursor)

    max = 0
    most_freq_word_id = -1
    for key in keys:
        current_word_frequency = count_word(current_user, key_words[key])
        if current_word_frequency >= max:
            max = current_word_frequency
            most_freq_word_id = key

    if max == 0:
        print('Could not find anything, adding special present for user_id=' + str(current_user['id']))
        assign_present(cursor, current_user['id'], 20)
        add_prediction(cursor, current_user['id'], 20)
    else:
        # print('https://vk.com/id' + str(current_user['id']), key_words[most_freq_word_id])
        assign_present(cursor, current_user['id'], most_freq_word_id)
        add_prediction(cursor, current_user['id'], most_freq_word_id)


# drop_presents(cursor)
for i in range(0, 2000):
    if i % 100 == 0:
        print(str(i) + ' - ' + str(100 * i / 2000) + '%')
    mark_next_free_person()


db.commit()
db.close()
