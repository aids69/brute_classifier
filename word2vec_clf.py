import nltk
import re

from gensim.models import KeyedVectors
from nltk.tag import pos_tag, map_tag
from ufal.udpipe import Model, Pipeline


model = KeyedVectors.load_word2vec_format('/home/ftlka/Documents/diploma/brute_classifier/models/wiki_model.bin', binary=True)
modelfile = '/home/ftlka/Documents/diploma/brute_classifier/models/udpipe_syntagrus.model'

clusters = {
    1: ['мужчина_NOUN', 'армия_NOUN', 'кальян_NOUN', 'рыбалка_NOUN', 'футбол_NOUN',
        'man_NOUN', 'army_NOUN', 'fishing_NOUN', 'football_NOUN'],

    2: ['кот_NOUN', 'собака_NOUN', 'бездомный_ADJ', 'животное_NOUN',
        'cat_NOUN', 'dog_NOUN', 'shelter_NOUN', 'animal_NOUN'],

    3: ['мем_NOUN', 'шутка_NOUN', 'юмор_NOUN', 'гиф_NOUN',
        'meme_NOUN', 'joke_NOUN', 'humor_NOUN'],

    4: ['рецепт_NOUN', 'кулинарный_ADJ', 'рукоделие_NOUN',
        'recipe_NOUN', 'cook_NOUN', 'handmade_NOUN'],

    5: ['мультфильм_NOUN', 'морти_NOUN', 'настольный_ADJ', 'игра_NOUN',
        'cartoon_NOUN', 'game_NOUN'],

    6: ['спорт_NOUN', 'фитнес_NOUN', 'похудение_NOUN', 'баскетбол_NOUN',
        'sport_NOUN', 'fitness_NOUN', 'weight_NOUN', 'basketball_NOUN'],

    7: ['макияж_NOUN', 'модель_NOUN', 'модный_ADJ',
        'cosmetics_NOUN', 'model_NOUN', 'fashion_NOUN'],

    8: ['marvel_NOUN', 'фильм_NOUN', 'сериал_NOUN', 'комикс_NOUN',
        'movie_NOUN', 'series_NOUN', 'comics_NOUN', 'geek_NOUN'],

    9: ['кино_NOUN', 'режиссер_NOUN', 'сериал_NOUN', 'фильм_NOUN',
        'movie_NOUN', 'producer_NOUN', 'series_NOUN', 'film_NOUN'],

    10: ['музыка_NOUN', 'концерт_NOUN', 'радио_NOUN',
         'cover_NOUN', 'music_NOUN', 'concert_NOUN', 'radio_NOUN'],

    11: ['мама_NOUN', 'семья_NOUN', 'воспитание_NOUN',
         'mother_NOUN', 'family_NOUN', 'children_NOUN'],

    12: ['тату_NOUN', 'tattoo_NOUN', 'татуировка_NOUN'],

    13: ['университет_NOUN', 'студент_NOUN',
         'university_NOUN', 'student_NOUN'],

    14: ['английский_ADJ', 'язык_NOUN',
         'english_NOUN', 'language_NOUN'],

    15: ['аниме_NOUN', 'косплей_NOUN', 'anime_NOUN'],

    16: ['бизнес_NOUN', 'политика_NOUN', 'инвестирование_NOUN',
         'business_NOUN', 'politics_NOUN', 'money_NOUN'],

    17: ['компьютерный_ADJ', 'игра_NOUN', 'киберспорт_NOUN',
         'computer_NOUN', 'game_NOUN', 'esports_NOUN'],

    18: ['искусство_NOUN', 'творчество_NOUN', 'вдохновение_NOUN',
         'art_NOUN', 'inspiration_NOUN'],

    19: ['свадьба_NOUN', 'беременность_NOUN', 'wedding_NOUN', 'bride_NOUN'],

    21: ['наука_NOUN', 'прогресс_NOUN', 'интеллект_NOUN',
         'science_NOUN', 'progress_NOUN', 'intellect_NOUN'],

    22: ['машина_NOUN', 'дорога_NOUN', 'car_NOUN', 'road_NOUN'],

    23: ['программирование_NOUN', 'математика_NOUN', 'тестирование_NOUN',
         'programming_NOUN', 'math_NOUN', 'testing_NOUN'],

    24: ['книга_NOUN', 'чтение_NOUN', 'литература_NOUN',
         'book_NOUN', 'reading_NOUN', 'literature_NOUN'],

    25: ['ресторан_NOUN', 'мероприятие_NOUN', 'бар_NOUN',
         'restaurant_NOUN', 'event_NOUN', 'bar_NOUN'],

    26: ['йога_NOUN', 'танец_NOUN', 'единоборство_NOUN',
         'yoga_NOUN', 'dance_NOUN', 'combat_NOUN'],

    27: ['путешествие_NOUN', 'туризм_NOUN', 'авиабилет_NOUN',
         'journey_NOUN', 'tourism_NOUN'],

    28: ['девушка_NOUN', 'маникюр_NOUN', 'прическа_NOUN', 'танец_NOUN',
         'girl_NOUN', 'nails_NOUN', 'hair_NOUN', 'dance_NOUN'],

    29: ['миф_NOUN', 'астрология_NOUN', 'квест_NOUN', 'ведьма_NOUN', 'зодиак_NOUN',
         'myth_NOUN', 'astrology_NOUN', 'quest_NOUN', 'witch_NOUN', 'zodiac_NOUN'],

    30: ['театр_NOUN', 'эстрада_NOUN', 'актер_NOUN',
         'theatre_NOUN', 'stage_NOUN', 'actor_NOUN'],

    31: ['дизайн_NOUN', 'ремонт_NOUN', 'интерьер_NOUN', 'design_NOUN'],

    32: ['велосипед_NOUN', 'bicycle_NOUN'],

    33: ['активный_ADJ', 'поход_NOUN', 'сноуборд_NOUN', 'альпинизм_NOUN',
         'snowboard_NOUN']
}


def _tag_ud(text):
    """Formats input string to required format for our pre-trained word2vec model"""
    model = Model.load(modelfile)
    pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
    processed = pipeline.process(text)

    output = [l for l in processed.split('\n') if not l.startswith('#')]
    tagged = [w.split('\t')[2].lower() + '_' + w.split('\t')[3] for w in output if w]

    tagged_propn = []
    propn = []
    for t in tagged:
        if t.endswith('PROPN'):
            if propn:
                propn.append(t)
            else:
                propn = [t]
        else:
            if len(propn) > 1:
                name = '::'.join([x.split('_')[0] for x in propn]) + '_PROPN'
                tagged_propn.append(name)
            elif len(propn) == 1:
                tagged_propn.append(propn[0])
            tagged_propn.append(t)
            propn = []

    return tagged_propn


def _filter_model_words(current_words):
    """Filters out words that are not in our vocabulary"""
    return [word for word in current_words if word in model.wv.vocab]


def _get_english_lexemes(sentence):
    """Creates lexemes for english words"""
    sentence = ' '.join(sentence)
    text = nltk.word_tokenize(sentence)
    posTagged = pos_tag(text)

    simplifiedTags = [(word, map_tag('en-ptb', 'universal', tag)) for word, tag in posTagged]
    return simplifiedTags


def find_most_similar_class(user_info):
    """Find most appropriate class for current userinfo"""
    russian_lexems = _tag_ud(user_info)
    english_lexemes = _get_english_lexemes([word for word in user_info.split(' ') if re.match(r'[A-Za-z]', word)])
    english_lexemes = ['_'.join(el) for el in english_lexemes]

    lexemes = _filter_model_words(russian_lexems + english_lexemes)

    evaluation = []
    for key, cluster in clusters.items():
        current_value = model.wv.n_similarity(lexemes, cluster)
        evaluation.append((key, current_value, cluster))
    evaluation.sort(key=lambda x: x[1], reverse=True)

    for i, val in zip(range(7), evaluation):
        print(val)

    return evaluation[0][0], evaluation[0][1], evaluation[0][2]
