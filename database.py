import sqlite3

DB_NAME = "jarvis.db"


def get_connection():
    return sqlite3.connect(DB_NAME)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        local_id INTEGER NOT NULL,
        task TEXT NOT NULL,
        status TEXT DEFAULT 'pending',
        UNIQUE(session_id, local_id)
    )
""")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            note TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()