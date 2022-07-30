import sqlite3

DB_CON = sqlite3.connect("./main.db")
DB_CUR = DB_CON.cursor()

DB_CUR.execute("INSERT INTO images (img) VALUES ('dd')")