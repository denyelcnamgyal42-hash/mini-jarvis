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
        "SELECT COALESCE(MAX(local_id), 0) + 1 FROM tasks WHERE session_id = ?",
        (current_session_id,)
    )

    next_local_id = cursor.fetchone()[0]

    cursor.execute(
        "INSERT INTO tasks (session_id, local_id, task, status) VALUES (?, ?, ?, ?)",
        (current_session_id, next_local_id, task, "pending")
    )

    conn.commit()
    conn.close()

    return f"Added task {next_local_id}: {task}"


@tool
def list_tasks() -> str:
    """List tasks."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT local_id, task, status
        FROM tasks
        WHERE session_id = ?
        ORDER BY local_id
        """,
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
        """
        UPDATE tasks
        SET status = ?
        WHERE local_id = ? AND session_id = ?
        """,
        ("done", task_id, current_session_id)
    )

    conn.commit()

    if cursor.rowcount == 0:
        conn.close()
        return f"No task found with ID {task_id}."

    conn.close()
    return f"Task {task_id} completed."


@tool
def delete_task(task_id: int) -> str:
    """Delete task."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM tasks
        WHERE local_id = ? AND session_id = ?
        """,
        (task_id, current_session_id)
    )

    conn.commit()

    if cursor.rowcount == 0:
        conn.close()
        return f"No task found with ID {task_id}."

    conn.close()
    return f"Task {task_id} deleted."


@tool
def update_task(task_id: int, new_task: str) -> str:
    """Update task."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE tasks
        SET task = ?
        WHERE local_id = ? AND session_id = ?
        """,
        (new_task, task_id, current_session_id)
    )

    conn.commit()

    if cursor.rowcount == 0:
        conn.close()
        return f"No task found with ID {task_id}."

    conn.close()
    return f"Task {task_id} updated to: {new_task}"


@tool
def clear_completed_tasks() -> str:
    """Clear completed tasks."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM tasks
        WHERE status = ? AND session_id = ?
        """,
        ("done", current_session_id)
    )

    deleted_count = cursor.rowcount

    conn.commit()
    conn.close()

    return f"Cleared {deleted_count} completed task(s)."


@tool
def get_today_focus() -> str:
    """Get focus task."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT local_id, task
        FROM tasks
        WHERE session_id = ? AND status = ?
        ORDER BY local_id
        LIMIT 1
        """,
        (current_session_id, "pending")
    )

    row = cursor.fetchone()
    conn.close()

    if not row:
        return "You have no pending tasks. Good job."

    return f"Today's focus: Task {row[0]} - {row[1]}"


@tool
def get_task_summary() -> str:
    """Summarize tasks."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT status, COUNT(*)
        FROM tasks
        WHERE session_id = ?
        GROUP BY status
        """,
        (current_session_id,)
    )

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return "You have no tasks yet."

    summary = {
        "pending": 0,
        "done": 0
    }

    for status, count in rows:
        summary[status] = count

    return (
        f"Task summary: "
        f"{summary['pending']} pending, "
        f"{summary['done']} completed."
    )


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
    delete_task,
    update_task,
    clear_completed_tasks,
    get_today_focus,
    get_task_summary,
    save_note,
    list_notes,
    get_current_time
]