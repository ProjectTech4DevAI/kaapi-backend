from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

ChatbotMessageDict = dict[str, str]


class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
