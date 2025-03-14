import sqlite3
import pandas as pd
from datetime import datetime

# Initialize Database
def init_db():
    conn = sqlite3.connect('data_genius.db')
    c = conn.cursor()

    # Create users table
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        registration_date TEXT
    )
    ''')

    # Create insights table
    c.execute('''
    CREATE TABLE IF NOT EXISTS insights (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        insights_text TEXT,
        date_created TEXT,
        FOREIGN KEY (username) REFERENCES users (username)
    )
    ''')

    # Create visualizations table
    c.execute('''
    CREATE TABLE IF NOT EXISTS visualizations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        image BLOB,
        date_created TEXT,
        FOREIGN KEY (username) REFERENCES users (username)
    )
    ''')

    conn.commit()
    conn.close()

# Save new user
def save_user(username, password):
    conn = sqlite3.connect('data_genius.db')
    c = conn.cursor()
    try:
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO users (username, password, registration_date) VALUES (?, ?, ?)", 
                  (username, password, current_date))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Username already exists
    finally:
        conn.close()

# Check user login
def check_user(username, password):
    conn = sqlite3.connect('data_genius.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = c.fetchone()
    conn.close()
    return user is not None

# Save insights to database
def save_insights(username, insights_text):
    conn = sqlite3.connect('data_genius.db')
    c = conn.cursor()
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO insights (username, insights_text, date_created) VALUES (?, ?, ?)", 
              (username, insights_text, current_date))
    conn.commit()
    conn.close()

# Save visualization to database
def save_visualization(username, image_bytes):
    conn = sqlite3.connect('data_genius.db')
    c = conn.cursor()
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO visualizations (username, image, date_created) VALUES (?, ?, ?)", 
              (username, image_bytes, current_date))
    conn.commit()
    conn.close()

# Fetch all data for admin panel
def get_all_data():
    conn = sqlite3.connect('data_genius.db')
    
    # Get users data
    users_df = pd.read_sql_query("SELECT * FROM users", conn)
    
    # Get insights data
    insights_df = pd.read_sql_query("SELECT * FROM insights", conn)
    
    # Get visualizations data (excluding the actual images for display purposes)
    visualizations_df = pd.read_sql_query("SELECT id, username, date_created FROM visualizations", conn)
    
    conn.close()
    
    return users_df, insights_df, visualizations_df

# Initialize database on first run
init_db()


import os

CSV_FILE = "session_insights.csv"

def save_insights_to_csv(username, insights_text):
    # Check if file exists
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    else:
        df = pd.DataFrame(columns=["username", "insights_text"])

    # Add new insight
    new_data = pd.DataFrame([[username, insights_text]], columns=["username", "insights_text"])
    df = pd.concat([df, new_data], ignore_index=True)

    # Save to CSV
    df.to_csv(CSV_FILE, index=False)
