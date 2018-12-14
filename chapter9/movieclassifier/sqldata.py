import sqlite3
import os

def create_db():
    conn = sqlite3.connect('reviews.sqlite')
    c = conn.cursor()
    c.execute('create table review_db'\
              '(review TEXT, sentiment INTEGER, date TEXT)')
    example1 = 'I love this movie'
    c.execute("INSERT INTO review_db"\
              "(review, sentiment, date) VALUES "\
              "(?, ?, DATETIME('now'))", (example1, 1))
    example2 = 'I disliked this movie'
    c.execute("INSERT INTO review_db"\
              "(review, sentiment,date) VALUES "\
              "(?, ?, DATETIME('now'))", (example2, 0))
    conn.commit()
    conn.close()

if __name__ == '__main__':
    conn = sqlite3.connect('reviews.sqlite')
    c = conn.cursor()
    c.execute("SELECT * FROM review_db WHERE date BETWEEN '2018-01-01 00:00:00' AND DATETIME('now')")
    result = c.fetchall()
    conn.close()
    print(result)