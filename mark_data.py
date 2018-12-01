import sqlite3
from db_api import get_key_words, get_user, assign_present


db = sqlite3.connect('db/users.db')
cursor = db.cursor()

word_dict = get_key_words(cursor)



db.commit()
db.close()
