from typing import Annotated, TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv

from tools import jarvis_tools, set_session_id

load_dotenv()

chat_histories = {}

class JarvisState(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatGroq(
    model = 'llama-3.1-8b-instant',
    temperature=0,
    max_tokens=150, 
    timeout=30
)

llm_with_tools = llm.bind_tools(jarvis_tools)

def jarvis_node(state: JarvisState):
    system_prompt = SystemMessage(
        content = (
        "You are Mini Jarvis. Use tools directly. "
        "For adding tasks, call add_task. "
        "For listing tasks, call list_tasks. "
        "For completing tasks, call complete_task. "
        "Reply briefly."
        )
    )

    messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages)

    return {"messages":[response]}

graph_builder = StateGraph(JarvisState)

graph_builder.add_node("jarvis", jarvis_node)
graph_builder.add_node("tools", ToolNode(jarvis_tools))

graph_builder.add_edge(START, "jarvis")
graph_builder.add_conditional_edges("jarvis", tools_condition)
graph_builder.add_edge("tools", END)

jarvis_graph = graph_builder.compile()

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