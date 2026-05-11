from datetime import datetime
from typing import Annotated, Literal, TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import jarvis_tools, set_session_id
from database import search_memories_db, save_memory_db, memory_exists_db


# -----------------------------
# 1. Agent State
# -----------------------------

class JarvisState(TypedDict):
    messages: Annotated[list, add_messages]
    route: str
    memory_context: str

# -----------------------------
# 2. LLM Setup
# -----------------------------

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=500,
    timeout=30
)

llm_with_tools = llm.bind_tools(jarvis_tools)


# -----------------------------
# 3. Helper Function
# -----------------------------

def get_last_user_message(state: JarvisState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.content
    return ""


# -----------------------------
# 4. Router Node
# -----------------------------

def router_node(state: JarvisState):
    """
    The router decides which skill should handle the user's message.

    This is not doing the final work.
    It only chooses the next node.
    """

    user_message = get_last_user_message(state)

    system_prompt = SystemMessage(
        content=(
            "You are the routing brain of Mini Jarvis. "
            "Classify the user's message into exactly one route. "
            "Return only one word from this list:\n"
            "task_agent\n"
            "web_research\n"
            "coding_helper\n"
            "planning_helper\n"
            "planner_agent\n"
            "general_chat\n\n"
            "Rules:\n"
            "- Use task_agent for tasks, reminders, notes, memories, remember requests, questions like what do you remember about me, what do I like, what are my preferences, due dates, listing tasks, completing tasks, dashboard-like requests.\n"
            "- Use web_research for current/latest/recent/news/search/online information, or when the user asks about a person, footballer, player, company, team, event, or unknown named entity.\n"
            "- Use coding_helper for code, errors, debugging, stack traces, terminal logs, Git/GitHub issues, deployment errors, APIs, frameworks, package installation, backend/frontend bugs, or programming explanations.\n"
            "- Use planning_helper for study plans, project plans, schedules, roadmaps, step-by-step plans.\n"
            "- Use planner_agent for broad goals that require multiple steps, such as preparing for exams, completing projects, building systems, researching then planning, or organizing work across multiple days.\n"
            "- Use general_chat for explanations, normal questions, casual chat, or anything else."
        )
    )

    response = llm.invoke([
        system_prompt,
        HumanMessage(content=user_message)
    ])

    route = response.content.strip().lower()

    allowed_routes = {
        "task_agent",
        "web_research",
        "coding_helper",
        "planning_helper",
        "planner_agent",
        "general_chat"
    }

    if route not in allowed_routes:
        route = "general_chat"

    print("Router selected:", route)

    return {"route": route}


def route_to_skill(state: JarvisState) -> Literal[
    "task_agent",
    "web_research",
    "coding_helper",
    "planning_helper",
    "planenr_agent",
    "general_chat"
]:
    return state["route"]


# -----------------------------
# 5. General Chat Skill
# -----------------------------

def general_chat_node(state: JarvisState):
    memory_context = state.get("memory_context", "No relevant long-term memories.")

    system_prompt = SystemMessage(
        content=(
            "You are Mini Jarvis, a helpful AI assistant. "
            "Answer clearly and concisely. "
            "Do not claim to perform actions unless a tool was used.\n\n"
            f"Relevant long-term memory:\n{memory_context}"
        )
    )

    response = llm.invoke([system_prompt] + state["messages"])

    return {"messages": [response]}

# -----------------------------
# 6. Coding Helper Skill
# -----------------------------

def coding_helper_node(state: JarvisState):
    memory_context = state.get("memory_context", "No relevant long-term memories.")
    system_prompt = SystemMessage(
        content=(
            "You are Mini Jarvis in coding/debugging mode. "
            "Your job is to help the user understand and fix programming problems. "
            "The user is learning, so teach clearly while solving the issue. "
            "\n\n"
            "Follow this response structure when debugging:\n"
            "1. Problem found: briefly identify the main issue.\n"
            "2. Why it happens: explain the cause in simple technical terms.\n"
            "3. Fix: give the exact code or command needed.\n"
            "4. Test: tell the user how to verify it worked.\n"
            "\n\n"
            "Rules:\n"
            "- Do not pretend you ran the code.\n"
            "- Do not invent files unless the user mentions them.\n"
            "- If the error is clearly shown, do not ask for more details first; give the likely fix.\n"
            "- For terminal commands, prefer Windows PowerShell commands unless the user says otherwise.\n"
            "- For dangerous commands such as deleting files, clearly explain what they do.\n"
            "- Keep answers practical and concise.\n"
            "- If the issue is related to this Jarvis project, connect the explanation to FastAPI, LangGraph, tools, routes, or frontend/backend flow."
            f"\n\nRelevant long-term memory:\n{memory_context}"
        )
    )

    response = llm.invoke([system_prompt] + state["messages"])

    return {"messages": [response]}

# -----------------------------
# 7. Planning Helper Skill
# -----------------------------

def planning_helper_node(state: JarvisState):
    memory_context = state.get("memory_context", "No relevant long-term memories.")
    system_prompt = SystemMessage(
        content=(
            "You are Mini Jarvis in planning mode. "
            "Create practical, structured plans. "
            "Break work into clear phases and next actions. "
            "Avoid vague advice. "
            "If the user asks for a schedule, make it realistic."
            f"\n\nRelevant long-term memory:\n{memory_context}"
        )
    )

    response = llm.invoke([system_prompt] + state["messages"])

    return {"messages": [response]}

def planner_agent_node(state: JarvisState):
    """
    Executable multi-step planner skill.

    This node can reason about broad goals and call safe task tools
    to create tasks when useful.
    """

    today = datetime.now().date().isoformat()

    safe_planner_tools = [
        tool for tool in jarvis_tools
        if tool.name in [
            "add_task",
            "list_tasks",
            "get_today_focus",
            "get_task_summary"
        ]
    ]

    planner_llm_with_tools = llm.bind_tools(safe_planner_tools)

    system_prompt = SystemMessage(
        content=(
            "You are Mini Jarvis in executable planner mode. "
            f"Today's date is {today}. "
            "Your job is to help the user turn a broad goal into an actionable plan. "
            "\n\n"
            "You may use only these tools: add_task, list_tasks, get_today_focus, get_task_summary. "
            "Use add_task when the user clearly wants help organizing work, preparing, studying, or completing a project. "
            "When creating tasks, make them short and practical. "
            "If the user mentions a deadline like today, tomorrow, Friday, or a date, pass due_date as YYYY-MM-DD when possible. "
            "\n\n"
            "Important rules:\n"
            "- Do not delete, update, or complete tasks in planner mode.\n"
            "- Do not create too many tasks. Usually create 3 to 5 tasks maximum.\n"
            "- If the user only asks for advice and not organization, give a plan without tools.\n"
            "- If you call tools, the tool result will be returned first.\n"
            "- Be practical and concise."
        )
    )

    response = planner_llm_with_tools.invoke([system_prompt] + state["messages"])

    return {"messages": [response]}

def get_memory_context(session_id: str, user_message: str) -> str:
    """
    Retrieve relevant long-term memories for this user/session.
    """

    rows = search_memories_db(session_id, user_message)

    if not rows:
        return "No relevant long-term memories."

    memory_lines = [
        f"- {row[1]}"
        for row in rows
    ]

    return "\n".join(memory_lines)

def extract_and_save_memory(
    user_message: str,
    assistant_reply: str,
    session_id: str,
    recent_history: list
):
    """
    Automatically extracts useful long-term memories from the recent conversation.
    """

    history_text = ""

    for msg in recent_history[-6:]:
        role = msg.__class__.__name__
        content = getattr(msg, "content", "")
        if content:
            history_text += f"{role}: {content}\n"

    system_prompt = SystemMessage(
        content=(
            "You are a memory extraction module for Mini Jarvis. "
            "Your job is to extract useful long-term memories about the user from the conversation. "
            "\n\n"
            "Save memories about:\n"
            "- User preferences and likes\n"
            "- User interests\n"
            "- User goals\n"
            "- Projects the user is working on\n"
            "- Tools, frameworks, or technologies the user uses\n"
            "- Stable personal context that can improve future help\n"
            "Do not save memories from web research questions unless the user clearly says they personally like, prefer, play, use, study, or are working on something. "
            "Do not save facts about public figures as user memories. "
            "\n\n"
            "Important:\n"
            "- Use the recent conversation to resolve references like 'him', 'that', or 'yeah'. "
            "- Correct obvious spelling mistakes when the meaning is clear. "
            "- If the user says they like someone/something, save it. "
            "- Return only one memory at a time. "
            "\n\n"
            "Do NOT save:\n"
            "- Temporary conversation details\n"
            "- Assistant guesses that the user did not confirm\n"
            "- Sensitive personal data unless the user clearly asks to remember it\n"
            "\n\n"
            "Return only one of these formats:\n"
            "NO_MEMORY\n"
            "MEMORY: <one clear sentence about the user>\n\n"
            "Examples:\n"
            "Conversation: User says they like football.\n"
            "MEMORY: User likes football.\n\n"
            "Conversation: User says they play as a midfielder.\n"
            "MEMORY: User plays as a midfielder in football.\n\n"
            "Conversation: User says they like Kevin De Bruyne.\n"
            "MEMORY: User likes Kevin De Bruyne.\n\n"
            "Conversation: User says okay thanks.\n"
            "NO_MEMORY"
        )
    )

    user_prompt = HumanMessage(
        content=(
            f"Recent conversation:\n{history_text}\n\n"
            f"Latest user message:\n{user_message}\n\n"
            f"Latest assistant reply:\n{assistant_reply}"
        )
    )

    try:
        response = llm.invoke([system_prompt, user_prompt])
        result = response.content.strip()

        print("agent.py: memory extractor result:", result)

        if result.startswith("MEMORY:"):
            memory = result.replace("MEMORY:", "", 1).strip()

            if memory:
                if memory_exists_db(session_id, memory):
                    print("agent.py: duplicate memory skipped:", memory)
                else:
                    save_memory_db(session_id, memory)
                    print("agent.py: saved memory:", memory)

    except Exception as e:
        print("agent.py: memory extraction error:", e)

def planner_summary_node(state: JarvisState):
    """
    Final response after planner has used tools.

    This explains what was created and gives the user a clear next step.
    """

    system_prompt = SystemMessage(
        content=(
            "You are Mini Jarvis. The planner has just used tools to organize the user's goal. "
            "Now give a concise final response. "
            "\n\n"
            "Include:\n"
            "1. What tasks were created or checked\n"
            "2. A short action plan\n"
            "3. The immediate next step\n"
            "\n\n"
            "Do not claim a task was created unless the tool messages show it was created."
        )
    )

    response = llm.invoke([system_prompt] + state["messages"])

    return {"messages": [response]}

# -----------------------------
# 8. Task Agent Skill
# -----------------------------

def task_agent_node(state: JarvisState):
    today = datetime.now().date().isoformat()

    system_prompt = SystemMessage(
    content=(
        f"You are Mini Jarvis in task/reminder/memory mode. "
        f"Today's date is {today}. "
        "You may only use these tools: add_task, list_tasks, complete_task, "
        "delete_task, update_task, clear_completed_tasks, list_today_tasks, "
        "list_overdue_tasks, get_today_focus, get_task_summary, save_note, "
        "list_notes, get_current_time, save_memory, list_memories, search_memories. "
        "Do not use web_search in this mode. "
        "Use tools for task, note, reminder, due date, time, and memory requests. "
        "If the user says remember, memorize, save this about me, use save_memory. "
        "If the user asks what you remember, what they like, or what their preferences are, use list_memories or search_memories. "
        "When the user gives a due date like today, tomorrow, Friday, or a date, "
        "pass the due_date argument as YYYY-MM-DD when possible. "
        "Do not call any tool that is not listed. "
        "Be concise."
    )
)

    response = llm_with_tools.invoke([system_prompt] + state["messages"])

    return {"messages": [response]}


# -----------------------------
# 9. Web Research Skill
# -----------------------------

def web_research_node(state: JarvisState):
    memory_context = state.get("memory_context", "No relevant long-term memories.")

    system_prompt = SystemMessage(
        content=(
            "You are Mini Jarvis in web research mode. "
            "Your job is to answer questions that require current or online information. "
            "\n\n"
            "Use only the web_search tool. "
            "Do not answer from memory when the user asks for latest, today, recent, current, match results, lineups, transfers, injuries, news, or live/current information. "
            "\n\n"
            "When the user's question depends on previous context, include that context in the search query. "
            "For example, if the user previously discussed Lamine Yamal and then asks 'did he play?', search for the specific player and match, not just 'did he play'. "
            "\n\n"
            "For sports questions, search with specific terms: team names, player name, score, competition, and date if known. "
            "Prefer official club pages, match reports, Reuters, AP, ESPN, BBC, Sky Sports, FotMob, or reliable sports outlets. "
            "\n\n"
            "If search results conflict, say that results conflict and explain which source you trust more. "
            "Do not invent goals, assists, lineups, or scores. "
            "\n\n"
            f"Relevant long-term memory:\n{memory_context}"
        )
    )

    response = llm_with_tools.invoke([system_prompt] + state["messages"])

    return {"messages": [response]}


# -----------------------------
# 10. Route After Tool Use
# -----------------------------

def route_after_skill(state: JarvisState):
    """
    After task_agent or web_research runs, check if the model requested a tool.
    If yes, go to tools.
    If no, end.
    """

    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return "end"


def route_after_tools(state: JarvisState):
    """
    Decide what happens after tools run.

    - web_search should be summarized.
    - planner_agent tool use should be followed by a planner summary.
    - task tools should end directly.
    """

    last_message = state["messages"][-1]

    if isinstance(last_message, ToolMessage) and last_message.name == "web_search":
        return "summarize_search"

    if state.get("route") == "planner_agent":
        return "planner_summary"

    return "end"


# -----------------------------
# 11. Search Summarizer Skill
# -----------------------------

def summarize_search_node(state: JarvisState):
    system_prompt = SystemMessage(
        content=(
            "You are Mini Jarvis. The previous message contains web search results. "
            "Your job is to answer using only those search results. "
            "\n\n"
            "Rules:\n"
            "- Do not invent facts not present in the search results.\n"
            "- For sports results, clearly state the score, scorers, whether the player played, and the source basis if available.\n"
            "- If the search results say a player was missing, injured, suspended, or absent, say they did not play.\n"
            "- If the search results do not clearly answer the question, say you could not confirm it.\n"
            "- If sources conflict, mention the conflict instead of guessing.\n"
            "- Keep the answer concise."
        )
    )

    response = llm.invoke([system_prompt] + state["messages"])

    return {"messages": [response]}
# -----------------------------
# 12. Build LangGraph
# -----------------------------

graph_builder = StateGraph(JarvisState)

graph_builder.add_node("router", router_node)
graph_builder.add_node("general_chat", general_chat_node)
graph_builder.add_node("coding_helper", coding_helper_node)
graph_builder.add_node("planning_helper", planning_helper_node)
graph_builder.add_node("planner_agent", planner_agent_node)
graph_builder.add_node("planner_summary", planner_summary_node)
graph_builder.add_node("task_agent", task_agent_node)
graph_builder.add_node("web_research", web_research_node)
graph_builder.add_node("tools", ToolNode(jarvis_tools))
graph_builder.add_node("summarize_search", summarize_search_node)

graph_builder.add_edge(START, "router")

graph_builder.add_conditional_edges(
    "router",
    route_to_skill,
    {
        "task_agent": "task_agent",
        "web_research": "web_research",
        "coding_helper": "coding_helper",
        "planning_helper": "planning_helper",
        "planner_agent": "planner_agent",
        "general_chat": "general_chat"
    }
)

graph_builder.add_conditional_edges(
    "task_agent",
    route_after_skill,
    {
        "tools": "tools",
        "end": END
    }
)

graph_builder.add_conditional_edges(
    "web_research",
    route_after_skill,
    {
        "tools": "tools",
        "end": END
    }
)

graph_builder.add_edge("coding_helper", END)
graph_builder.add_edge("planning_helper", END)

graph_builder.add_conditional_edges(
    "planner_agent",
    route_after_skill,
    {
        "tools": "tools",
        "end": END
    }
)

graph_builder.add_edge("general_chat", END)

graph_builder.add_conditional_edges(
    "tools",
    route_after_tools,
    {
        "summarize_search": "summarize_search",
        "planner_summary": "planner_summary",
        "end": END
    }
)

graph_builder.add_edge("summarize_search", END)
graph_builder.add_edge("planner_summary", END)

jarvis_graph = graph_builder.compile()


# -----------------------------
# 13. Chat Memory Per Session
# -----------------------------

chat_histories = {}


def ask_jarvis(user_message: str, session_id: str) -> str:
    print("agent.py: received message")
    print("agent.py: session:", session_id)

    set_session_id(session_id)

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    chat_histories[session_id].append(HumanMessage(content=user_message))

    recent_history = chat_histories[session_id][-10:]

    memory_context = get_memory_context(session_id, user_message)

    print("agent.py: memory context:", memory_context)
    print("agent.py: before graph invoke")

    result = jarvis_graph.invoke(
        {
            "messages": recent_history,
            "route": "general_chat",
            "memory_context": memory_context
        },
        config={"recursion_limit": 10}
    )

    print("agent.py: after graph invoke")

    chat_histories[session_id] = result["messages"][-10:]

    final_message = result["messages"][-1]
    assistant_reply = final_message.content

    print("agent.py: final message ready")

    extract_and_save_memory(
        user_message=user_message,
        assistant_reply=assistant_reply,
        session_id=session_id,
        recent_history=chat_histories[session_id]
    )

    return assistant_reply

def detect_file_skill(filename: str) -> str:
    """
    Decide which file-analysis skill should handle the uploaded file.
    This is deterministic routing based on file extension.
    """

    name = filename.lower()

    frontend_extensions = (
        ".html", ".css", ".js", ".jsx", ".ts", ".tsx", ".vue"
    )

    backend_code_extensions = (
        ".py", ".java", ".cpp", ".c", ".go", ".rs", ".php", ".rb", ".cs"
    )

    document_extensions = (
        ".pdf", ".txt", ".md"
    )

    data_extensions = (
        ".json", ".csv", ".xml", ".yaml", ".yml"
    )

    if name.endswith(frontend_extensions):
        return "frontend_review"

    if name.endswith(backend_code_extensions):
        return "code_analysis"

    if name.endswith(document_extensions):
        return "document_analysis"

    if name.endswith(data_extensions):
        return "data_analysis"

    return "general_file_analysis"


def analyze_uploaded_file(
    filename: str,
    file_content: str,
    user_instruction: str,
    session_id: str
) -> str:
    """
    Routed file-analysis skill.

    The backend extracts text first.
    Then this function chooses the correct specialist prompt based on file type.
    """

    print("agent.py: analyzing uploaded file")
    print("agent.py: filename:", filename)
    print("agent.py: session:", session_id)

    file_skill = detect_file_skill(filename)

    print("agent.py: file skill:", file_skill)

    max_chars = 12000
    trimmed_content = file_content[:max_chars]

    if len(file_content) > max_chars:
        trimmed_content += "\n\n[File was shortened because it was too long.]"

    base_rules = (
        "You are Mini Jarvis. Analyze only the file content provided. "
        "Do not claim to see anything outside the uploaded file. "
        "If the user gives a specific instruction, follow it. "
        "Be clear, practical, and concise."
    )

    if file_skill == "code_analysis":
        skill_prompt = (
            "You are in backend/code analysis mode. "
            "Explain what the code does, identify bugs or risky parts, "
            "suggest improvements, and provide corrected code snippets if useful. "
            "Use this structure:\n"
            "1. What this file does\n"
            "2. Problems or risks\n"
            "3. Recommended fixes\n"
            "4. How to test"
        )

    elif file_skill == "frontend_review":
        skill_prompt = (
            "You are in frontend review mode. "
            "Analyze UI structure, JavaScript behavior, CSS quality, responsiveness, "
            "accessibility, and possible bugs. "
            "Use this structure:\n"
            "1. What this frontend file does\n"
            "2. UI/UX observations\n"
            "3. Bugs or issues\n"
            "4. Improvements\n"
            "5. Suggested code changes if needed"
        )

    elif file_skill == "document_analysis":
        skill_prompt = (
            "You are in document analysis mode. "
            "Summarize the document, extract key points, explain important concepts, "
            "and answer the user's instruction. "
            "Use this structure:\n"
            "1. Short summary\n"
            "2. Key points\n"
            "3. Important details\n"
            "4. Suggested next steps"
        )

    elif file_skill == "data_analysis":
        skill_prompt = (
            "You are in data/file structure analysis mode. "
            "Explain the structure of the data, identify important fields, "
            "spot inconsistencies if visible, and suggest how it could be used. "
            "Use this structure:\n"
            "1. Data structure\n"
            "2. Important fields or patterns\n"
            "3. Possible issues\n"
            "4. Suggested usage"
        )

    else:
        skill_prompt = (
            "You are in general file analysis mode. "
            "Explain what the file appears to contain and give useful observations."
        )

    system_prompt = SystemMessage(
        content=f"{base_rules}\n\n{skill_prompt}"
    )

    user_prompt = HumanMessage(
        content=(
            f"Filename: {filename}\n"
            f"Detected file skill: {file_skill}\n\n"
            f"User instruction: {user_instruction or 'Analyze this file.'}\n\n"
            f"File content:\n{trimmed_content}"
        )
    )

    response = llm.invoke([system_prompt, user_prompt])

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    chat_histories[session_id].append(
        HumanMessage(content=f"I uploaded {filename}. {user_instruction}")
    )
    chat_histories[session_id].append(response)
    chat_histories[session_id] = chat_histories[session_id][-10:]

    return response.content