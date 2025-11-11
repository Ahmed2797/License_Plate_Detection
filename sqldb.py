import sqlite3 

conn = sqlite3.connect('license_plate.db')
cursor = conn.cursor()

cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            starttime TEXT,
            endtime TEXT,
            license_plate TEXT
        )
    ''')
