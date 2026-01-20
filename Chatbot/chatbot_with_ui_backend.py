from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
class Chatstate(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

def chat_node(state: Chatstate) :
    messages=state["messages"]
    response = model.invoke(messages)
    return{"messages":[response]}

checkpointer = InMemorySaver()


graph = StateGraph(Chatstate)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)