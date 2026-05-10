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

def get_dashboard_data(session_id: str):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM tasks
        WHERE session_id = ? AND status = ?
        """,
        (session_id, "pending")
    )
    pending_count = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM tasks
        WHERE session_id = ? AND status = ?
        """,
        (session_id, "done")
    )
    completed_count = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT local_id, task
        FROM tasks
        WHERE session_id = ? AND status = ?
        ORDER BY local_id
        LIMIT 1
        """,
        (session_id, "pending")
    )
    focus_row = cursor.fetchone()

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM notes
        WHERE session_id = ?
        """,
        (session_id,)
    )
    notes_count = cursor.fetchone()[0]

    conn.close()

    today_focus = None

    if focus_row:
        today_focus = {
            "id": focus_row[0],
            "task": focus_row[1]
        }

    return {
        "pending_tasks": pending_count,
        "completed_tasks": completed_count,
        "notes_count": notes_count,
        "today_focus": today_focus
    }