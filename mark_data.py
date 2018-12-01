import sqlite3
import random
from db_api import get_key_words, get_user, assign_present, drop_presents

db = sqlite3.connect('db/users.db')
cursor = db.cursor()

key_words = get_key_words(cursor)


def count_word(user, words):
    """Counts substrings of words in user profile and can get maximum +1 count for each community"""
    counter = 0

    for comm in user['communities']:
        if any([s for s in comm if any(xs in s for xs in words)]):
            counter += 1

    informative_fields = ['about', 'activities', 'interests', 'inspired_by', 'status']
    for field in informative_fields:
        if field in user.keys():
            counter += sum(any(xs in s for xs in words) for s in user[field])

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
        print('Could not find anything, adding random present for user_id=' + str(current_user['id']))
    else:
        print('https://vk.com/id' + str(current_user['id']), key_words[most_freq_word_id])
        assign_present(cursor, current_user['id'], most_freq_word_id)


for i in range(0, 40):
    mark_next_free_person()
drop_presents(cursor)

db.commit()
db.close()
