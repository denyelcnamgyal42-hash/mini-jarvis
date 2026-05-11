import os
from datetime import datetime, timedelta
from langchain_core.tools import tool
from tavily import TavilyClient
from database import (
    get_connection,
    save_memory_db,
    list_memories_db,
    search_memories_db
)
from dotenv import load_dotenv

load_dotenv()


current_session_id = "default"
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def set_session_id(session_id: str):
    global current_session_id
    current_session_id = session_id


def normalize_due_date(due_date: str | None) -> str | None:
    """
    Converts simple natural date words into YYYY-MM-DD.
    The LLM should usually send YYYY-MM-DD, but this protects us.
    """

    if not due_date:
        return None

    value = due_date.strip().lower()
    today = datetime.now().date()

    if value in ["none", "no due date", ""]:
        return None

    if value == "today":
        return today.isoformat()

    if value == "tomorrow":
        return (today + timedelta(days=1)).isoformat()

    # If already ISO format, keep it
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return value
    except ValueError:
        pass

    # Simple weekday parsing
    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6
    }

    if value in weekdays:
        target_day = weekdays[value]
        days_ahead = target_day - today.weekday()

        if days_ahead <= 0:
            days_ahead += 7

        return (today + timedelta(days=days_ahead)).isoformat()

    return due_date

@tool
def save_memory(memory: str) -> str:
    """Save memory."""
    save_memory_db(current_session_id, memory)
    return f"Memory saved: {memory}"


@tool
def list_memories() -> str:
    """List memories."""
    rows = list_memories_db(current_session_id)

    if not rows:
        return "No memories saved yet."

    return "\n".join(
        f"{row[0]}. {row[1]} ({row[2]})"
        for row in rows
    )


@tool
def search_memories(query: str) -> str:
    """Search memories."""
    rows = search_memories_db(current_session_id, query)

    if not rows:
        return "No matching memories found."

    return "\n".join(
        f"{row[0]}. {row[1]} ({row[2]})"
        for row in rows
    )

@tool
def add_task(task: str, due_date: str = "") -> str:
    """Add task with optional due date."""
    conn = get_connection()
    cursor = conn.cursor()

    final_due_date = normalize_due_date(due_date)

    cursor.execute(
        "SELECT COALESCE(MAX(local_id), 0) + 1 FROM tasks WHERE session_id = ?",
        (current_session_id,)
    )

    next_local_id = cursor.fetchone()[0]

    cursor.execute(
        """
        INSERT INTO tasks (session_id, local_id, task, status, due_date)
        VALUES (?, ?, ?, ?, ?)
        """,
        (current_session_id, next_local_id, task, "pending", final_due_date)
    )

    conn.commit()
    conn.close()

    if final_due_date:
        return f"Added task {next_local_id}: {task} | Due: {final_due_date}"

    return f"Added task {next_local_id}: {task}"


@tool
def list_tasks() -> str:
    """List tasks."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT local_id, task, status, due_date
        FROM tasks
        WHERE session_id = ?
        ORDER BY 
            CASE WHEN due_date IS NULL OR due_date = '' THEN 1 ELSE 0 END,
            due_date,
            local_id
        """,
        (current_session_id,)
    )

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return "No tasks found."

    lines = []

    for local_id, task, status, due_date in rows:
        if due_date:
            lines.append(f"{local_id}. {task} - {status} | Due: {due_date}")
        else:
            lines.append(f"{local_id}. {task} - {status}")

    return "\n".join(lines)


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
def update_task(task_id: int, new_task: str, due_date: str = "") -> str:
    """Update task with optional due date."""
    conn = get_connection()
    cursor = conn.cursor()

    final_due_date = normalize_due_date(due_date)

    if final_due_date:
        cursor.execute(
            """
            UPDATE tasks
            SET task = ?, due_date = ?
            WHERE local_id = ? AND session_id = ?
            """,
            (new_task, final_due_date, task_id, current_session_id)
        )
    else:
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

    if final_due_date:
        return f"Task {task_id} updated to: {new_task} | Due: {final_due_date}"

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
def list_today_tasks() -> str:
    """List tasks due today."""
    today = datetime.now().date().isoformat()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT local_id, task, status
        FROM tasks
        WHERE session_id = ?
        AND status = ?
        AND due_date = ?
        ORDER BY local_id
        """,
        (current_session_id, "pending", today)
    )

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return "No pending tasks due today."

    return "\n".join(
        f"{row[0]}. {row[1]} - {row[2]} | Due: {today}"
        for row in rows
    )


@tool
def list_overdue_tasks() -> str:
    """List overdue tasks."""
    today = datetime.now().date().isoformat()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT local_id, task, due_date
        FROM tasks
        WHERE session_id = ?
        AND status = ?
        AND due_date IS NOT NULL
        AND due_date != ''
        AND due_date < ?
        ORDER BY due_date, local_id
        """,
        (current_session_id, "pending", today)
    )

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return "No overdue tasks."

    return "\n".join(
        f"{row[0]}. {row[1]} | Due: {row[2]}"
        for row in rows
    )


@tool
def get_today_focus() -> str:
    """Get focus task."""
    conn = get_connection()
    cursor = conn.cursor()

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
        (current_session_id, "pending")
    )

    row = cursor.fetchone()
    conn.close()

    if not row:
        return "You have no pending tasks. Good job."

    task_id, task, due_date = row

    if due_date:
        return f"Today's focus: Task {task_id} - {task} | Due: {due_date}"

    return f"Today's focus: Task {task_id} - {task}"


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

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM tasks
        WHERE session_id = ?
        AND status = ?
        AND due_date = date('now', 'localtime')
        """,
        (current_session_id, "pending")
    )
    due_today = cursor.fetchone()[0]

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
        (current_session_id, "pending")
    )
    overdue = cursor.fetchone()[0]

    conn.close()

    summary = {
        "pending": 0,
        "done": 0
    }

    for status, count in rows:
        summary[status] = count

    return (
        f"Task summary: {summary['pending']} pending, "
        f"{summary['done']} completed, "
        f"{due_today} due today, "
        f"{overdue} overdue."
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


@tool
def web_search(query: str) -> str:
    """Search web."""
    try:
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
            include_raw_content=False
        )

        answer = response.get("answer")
        results = response.get("results", [])

        output = []

        output.append(f"Search query: {query}")

        if answer:
            output.append(f"Search answer:\n{answer}")

        if results:
            output.append("Sources:")

            for index, item in enumerate(results, start=1):
                title = item.get("title", "No title")
                url = item.get("url", "")
                content = item.get("content", "")

                output.append(
                    f"[{index}] {title}\n"
                    f"URL: {url}\n"
                    f"Snippet: {content}"
                )

        if not output:
            return "No web results found."

        return "\n\n".join(output)

    except Exception as e:
        return f"Web search error: {str(e)}"

jarvis_tools = [
    add_task,
    list_tasks,
    complete_task,
    delete_task,
    update_task,
    clear_completed_tasks,
    list_today_tasks,
    list_overdue_tasks,
    get_today_focus,
    get_task_summary,
    save_note,
    list_notes,
    get_current_time,
    web_search,
    save_memory,
    list_memories,
    search_memories
]