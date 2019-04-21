import random
from .keywords import keywords


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
        com_info = []
        com_fields = ['name', 'decsription', 'status']
        for com_field in com_fields:
            if com_field in comm:
                com_info.append(comm[com_field])
        com_info = ' '.join(com_info).lower().split()

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


def brute(user, communities):
    max_freq, most_freq_present_id = 0, -1

    for key, value in keywords.items():
        cur_freq = _count_word(user, communities, value)
        if cur_freq >= max_freq:
            max_freq, most_freq_present_id = cur_freq, key

    return 20 if max == 0 else most_freq_present_id


def classifier(user, communities):
    return 3


def w2v(user, communities):
    return 3
