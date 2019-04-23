import random
import sqlite3

from db_api import get_user, assign_present, add_prediction
from keywords import keywords as key_words


def count_word(user, words):
    """Counts substrings of words in user profile and can get maximum +1 count for each community"""
    counter = 0

    # 1 - female, 2 - male
    if 'sex' in user:
        if user['sex'] == 1 and 'мужчин' in words:
            return 0
        elif user['sex'] == 2 and\
                len({'макияж', 'девочк', 'мам'}.intersection(words)):
            return 0

    for idx, comm in enumerate(user['communities']):
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
                break

    informative_fields = ['about', 'activities', 'interests', 'inspired_by', 'status']
    for field in informative_fields:
        if field in user:
            counter += 3 * sum(any(xs in s for xs in words) for s in user[field])

    # authors or movie names do not correlate with our key words so we add plus one for being filled
    half_informative_fields = {
        'books': 'книги', 'games': 'игры', 'movies': 'фильмы',
        'music': 'музыка', 'quotes': 'успех'
    }
    half_inf = []
    for key, val in half_informative_fields.items():
        if key in user and user[key]:
            half_inf.append(half_informative_fields[key])
    counter += sum(any(xs in s for xs in words) for s in half_inf)

    return counter


def mark_next_free_person(cursor, id=-1):
    # make key words' order random for cases with more than 1 max
    keys = list(key_words)
    random.shuffle(keys)

    current_user = get_user(cursor, id)

    max = 0
    most_freq_word_id = -1
    for key in keys:
        current_word_frequency = count_word(current_user, key_words[key])
        if current_word_frequency >= max:
            max = current_word_frequency
            most_freq_word_id = key

    if max == 0:
        print('Could not find anything, adding special present for user_id=' + str(current_user['id']))
        most_freq_word_id = 20
    assign_present(cursor, current_user['id'], most_freq_word_id)
    add_prediction(cursor, current_user['id'], most_freq_word_id)

    return most_freq_word_id


if __name__ == '__main__':
    db = sqlite3.connect('db/users.db')
    cursor = db.cursor()
    n = 5000

    for i in range(n):
        if i % 100 == 0:
            print('{} - {}%'.format(str(i), str(100 * i / n)))
        mark_next_free_person(cursor)

    db.commit()
    db.close()
