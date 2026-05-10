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
            due_date TEXT,
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

    # Migration: add due_date if old tasks table does not have it
    cursor.execute("PRAGMA table_info(tasks)")
    columns = [column[1] for column in cursor.fetchall()]

    if "due_date" not in columns:
        cursor.execute("ALTER TABLE tasks ADD COLUMN due_date TEXT")

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
        SELECT local_id, task, due_date
        FROM tasks
        WHERE session_id = ? AND status = ?
        ORDER BY 
            CASE WHEN due_date IS NULL OR due_date = '' THEN 1 ELSE 0 END,
            due_date,
            local_id
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

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM tasks
        WHERE session_id = ?
        AND status = ?
        AND due_date = date('now', 'localtime')
        """,
        (session_id, "pending")
    )
    due_today_count = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM tasks
        WHERE session_id = ?
        AND status = ?
        AND due_date IS NOT NULL
        AND due_date != ''
        AND due_date < date('now', 'localtime')
        """,
        (session_id, "pending")
    )
    overdue_count = cursor.fetchone()[0]

    conn.close()

    today_focus = None

    if focus_row:
        today_focus = {
            "id": focus_row[0],
            "task": focus_row[1],
            "due_date": focus_row[2]
        }

    return {
        "pending_tasks": pending_count,
        "completed_tasks": completed_count,
        "notes_count": notes_count,
        "due_today": due_today_count,
        "overdue": overdue_count,
        "today_focus": today_focus
    }