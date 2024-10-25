import sqlite3
import numpy as np

class DatabaseManager:
    def __init__(self, db_path='face_data.db'):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                age INTEGER,
                description TEXT,
                image_url TEXT,
                embedding BLOB
            )
        ''')
        self.conn.commit()

    def store_face_details(self, name, age, description, image_url, embedding):
        self.cursor.execute('''
            INSERT INTO users (name, age, description, image_url, embedding)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, age, description, image_url, embedding))
        self.conn.commit()

    def close(self):
        self.conn.close()
