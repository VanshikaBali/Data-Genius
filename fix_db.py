import sqlite3

conn = sqlite3.connect("data_genius.db")
cursor = conn.cursor()

# Users Table
cursor.execute('''CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    registration_date TEXT NOT NULL
)''')

# Insights Table
cursor.execute('''CREATE TABLE IF NOT EXISTS insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    insights_text TEXT NOT NULL,
    date_created TEXT NOT NULL,
    FOREIGN KEY (username) REFERENCES users (username)
)''')

# Visualizations Table
cursor.execute('''CREATE TABLE IF NOT EXISTS visualizations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    image BLOB NOT NULL,
    date_created TEXT NOT NULL,
    FOREIGN KEY (username) REFERENCES users (username)
)''')
#Analysis results
cursor.execute('''CREATE TABLE IF NOT EXISTS analysis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    model_name TEXT NOT NULL,
    accuracy FLOAT NOT NULL,
    date_created TEXT NOT NULL,
    FOREIGN KEY (username) REFERENCES users (username)
)''')



conn.commit()
conn.close()

print("✅ Database initialized successfully!")
