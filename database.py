import sqlite3

conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

# Users table
c.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT,
    password TEXT
)
""")

# Prediction history table
c.execute("""
CREATE TABLE IF NOT EXISTS history (
    username TEXT,
    prediction TEXT,
    probability REAL
)
""")

conn.commit()
