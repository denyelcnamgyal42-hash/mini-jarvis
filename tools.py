from datetime import datetime
from langchain_core.tools import tool
from database import get_connection


current_session_id = "default"


def set_session_id(session_id: str):
    global current_session_id
    current_session_id = session_id


@tool
def add_task(task: str) -> str:
    """Add task."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO tasks (session_id, task, status) VALUES (?, ?, ?)",
        (current_session_id, task, "pending")
    )

    conn.commit()
    conn.close()

    return f"Added task: {task}"


@tool
def list_tasks() -> str:
    """List tasks."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, task, status FROM tasks WHERE session_id = ?",
        (current_session_id,)
    )

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return "No tasks found."

    return "\n".join(
        f"{row[0]}. {row[1]} - {row[2]}"
        for row in rows
    )


@tool
def complete_task(task_id: int) -> str:
    """Complete task."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE tasks SET status = ? WHERE id = ? AND session_id = ?",
        ("done", task_id, current_session_id)
    )

    conn.commit()

    if cursor.rowcount == 0:
        conn.close()
        return f"No task found with ID {task_id}."

    conn.close()
    return f"Task {task_id} completed."


@tool
def save_note(note: str) -> str:
    """Save note."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO notes (session_id, note) VALUES (?, ?)",
        (current_session_id, note)
    )

    conn.commit()
    conn.close()

    return "Note saved."


@tool
def list_notes() -> str:
    """List notes."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, note FROM notes WHERE session_id = ?",
        (current_session_id,)
    )

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return "No notes found."

    return "\n".join(
        f"{row[0]}. {row[1]}"
        for row in rows
    )


@tool
def get_current_time() -> str:
    """Get time."""
    now = datetime.now()
    return now.strftime("%A, %d %B %Y, %I:%M %p")


jarvis_tools = [
    add_task,
    list_tasks,
    complete_task,
    save_note,
    list_notes,
    get_current_time
]