# Models for predicting presents

## How to start it
Either by using simple interface or by:
`python3 index.py vk_id`
where vk_id can be user's nickname or numeric id

## Description
The project itself consists of 3 parts:
- brute force classification
- classifying cluster vectors
- searching for closest class with word2vec

file mark_data.py contains brute classifier to semi-manually label data. It just searches for substrings and selects class that scored the most
clustering.py makes cluster vector and saves it to db. This cluster vector is made for most informative fields of the user's account
classifiers.py contains several classifiers and a way to retrieve cluster vectors from db. With labels made by brute
word2vec_clf.py uses pretrained model from https://rusvectores.org/ru/models/ and ranks all classes by similarity to user profile
index.py starts all other classifiers and prints the results of their work

## key words we use for marking data
1 - мужчин мужик брутал мужск арми бокс рукопаш воен джентельм кача барбер кальян рыбалк охот гараж футбол мяч

2 - кошк коты коти cat собак бездомн приют животн

3 - мем mlg юмор смех смеш прикол гиф gif анекдот humor прикол meme dank

4 - кондитер рецепт кулинар вкусн рукодел шить блюда вышивк кройк вязан cook handmade

5 - морти мульт гик geek adventure nerd cartoon nintendo capcom настол

6 - спорт фитнес похуд тело трениров физкультур body gym workout basket баскет гандбол плавани пловец гимнаст бодибилд мотивация хоккей

7 - макияж космети beauty парфюм fashion glamour model модель мода модн makeup визаж

8 - marvel dc комикс сериал фильм geek nerd

9 - кин фильм movie film режисс сериал

10 - музык аудио концерт music radio радио микс альбом jazz cover акустик гитар drum kpop кпоп

11 - мам семь семей родител брак беремен материн воспитан детск

12 - тату татуировк татуаж tattoo

13 - студен урфу универ urfu

14 - английск english

15 - аниме косплей наруто манг anime naruto cosplay manga

16 - бизнес успешн лидер lead инвест финанс торгов money прогноз ставк раскрутк стартап политик экономик навальн ройзм village tj медуза етв лентач образовач news город банк

17 - компьютер игр dota game киберспорт

18 - искусств живопис рисунок рисовани art худож рисун творч вдохнов

19 - свадьб wedding визаж свадеб декор молодож невест беремен

20 - special id for people with non-informative groups

21 - наука science прогресс интеллект обучени learn

22 - авто машин шины дороги дорогах ралли дтп внедорожн

23 - программ dev habr тестир code coding develop разраб математик math яндекс

24 - книг book книжн литератур

25 - ресторан пицц кофейн бар гастроном мероприят праздн

26 - йог yoga танц danc единоборств

27 - travel путешеств тури discover авиабилет иммигр турагенств туризм турист

28 - девочк девушк маникюр женск императриц шальн прическ наращ ресниц флорист стих танец танц beauty парфюм

29 - феи ведьм квест сказк сказоч миф цветы цветочн зодиак астролог гороскоп лунн эзотери

30 - театр выступлен эстрад актер актёр

31 - креатив дизайн design ремонт декор шрифт интерьер

32 - велосипед

33 - активн сноуборд альпини скалолаз страйкбол лазертаг gopro прогулк антикафе скалодром турклуб поход горнолыж лыж
