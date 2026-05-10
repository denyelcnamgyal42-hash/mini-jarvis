from typing import Annotated, TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv

from tools import jarvis_tools, set_session_id

load_dotenv()

# Agent State

class JarvisState(TypedDict):
    messages: Annotated[list, add_messages]

# llm setup 

llm = ChatGroq(
    model = 'llama-3.1-8b-instant',
    temperature=0,
    max_tokens=150, 
    timeout=30
)

llm_with_tools = llm.bind_tools(jarvis_tools)

# main jarvis node

def jarvis_node(state: JarvisState):
    system_prompt = SystemMessage(
        content = (
        "You are Mini Jarvis, a helpful personal AI assistant. "
        "You may only use these tools: add_task, list_tasks, complete_task, "
        "delete_task, update_task, clear_completed_tasks, get_today_focus, "
        "get_task_summary, save_note, list_notes, get_current_time, web_search. "
        "Use task and note tools for personal productivity requests. "
        "Use get_current_time for time questions. "
        "Use web_search only when the user asks for current, latest, recent, online, news, search, or unknown information. "
        "For stable general knowledge, answer directly without tools. "
        "Do not call any tool that is not listed. "
        "Be concise and direct."
        )
    )

    messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages)

    return {"messages":[response]}

# Tool Router

def route_after_jarvis(state: JarvisState):
    """This function checks what the LLM decided.
    If the llm did not call a tool:
        END
    
    If the LLM called any tool:
        go to tools
    """

    last_message = state['messages'][-1]
    if not getattr(last_message, 'tool_calls', None):
        return "end"
    
    return "tools"

def route_after_tools(state: JarvisState):
    """This function checks which tool just ran
    If the last tool result came from web_search:
        go to summarize_search 
    Otherwise:
        END"""
    
    last_message = state['messages'][-1]

    if isinstance(last_message, ToolMessage):
        if last_message.name == "web_search":
            return "summarize_search"
    
    return "end"

# Search Summarizer Node 

def summarize_search_node(state: JarvisState):
    """
    This node turns raw web search results into a clean answer

    It does not call tools.
    It only sumamrizes the search result. 
    """
    system_prompt = SystemMessage(
        content = (
            "You are Mini Jarvis. The previous message contains web search results. "
            "Summarize the results into a clear, concise answer. "
            "Include important points and mention sources briefly if URLs are available. "
            "Do not invent information beyond the search results."
        )
    )

    response = llm.invoke([system_prompt] + state["messages"])

    return {"messages" : [response]}

# Build Langgraph

graph_builder = StateGraph(JarvisState)

graph_builder.add_node("jarvis", jarvis_node)
graph_builder.add_node("tools", ToolNode(jarvis_tools))
graph_builder.add_node("summarize_search", summarize_search_node)

graph_builder.add_edge(START, "jarvis")

graph_builder.add_conditional_edges(
    "jarvis",
    route_after_jarvis,
    {
        "tools":"tools",
        "end": END
    }
)

graph_builder.add_conditional_edges(
    "tools",
    route_after_tools,
    {
        "summarize_search":"summarize_search",
        "end": END
    }
)

graph_builder.add_edge("summarize_search", END)

jarvis_graph = graph_builder.compile()

# Chat Memory Per Session
chat_histories = {}


def ask_jarvis(user_message: str, session_id: str) -> str:
    print("agent.py: received message")
    print("agent.py: session:", session_id)

    set_session_id(session_id)

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    chat_histories[session_id].append(HumanMessage(content=user_message))

    recent_history = chat_histories[session_id][-6:]

    print("agent.py: before graph invoke")

    result = jarvis_graph.invoke(
        {"messages": recent_history},
        config={"recursion_limit": 5}
    )

    print("agent.py: after graph invoke")

    chat_histories[session_id] = result["messages"][-6:]

    final_message = result["messages"][-1]

    print("agent.py: final message ready")

    return final_message.content