import os
import getpass
from typing import Sequence
from typing_extensions import Annotated, TypedDict

# LangChain imports
from langchain_openai import ChatOpenAI
# from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# LangGraph imports
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

class ChatApp:
    def __init__(self):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

        self.model = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000)
        

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        self.trimmer = trim_messages(
            max_tokens=50,
            strategy="last",
            token_counter=self.model,
            include_system=True,
            allow_partial=False,
            start_on="human",
        )

        self.workflow = StateGraph(state_schema=State)
        self.workflow.add_edge(START, "model")
        self.workflow.add_node("model", self.call_model)
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)
        self.config = {"configurable": {"thread_id": "abc123"}}
        self.messages = []

    def call_model(self, state: MessagesState):
        trimmed_messages = self.trimmer.invoke(state["messages"])
        prompt = self.prompt_template.invoke(
            {"messages": trimmed_messages, "language": state["language"]}
        )
        response = self.model.invoke(prompt)
        return {"messages": response}

    def add_message(self, message: str):
        self.messages.append(HumanMessage(message))

    def get_response(self):
        output = self.app.invoke({"messages": self.messages, "language": "Italian"}, self.config)
        return output["messages"][-1]

    def stream_response(self, input_messages, language, config):
        for chunk, metadata in self.app.stream(
            {"messages": input_messages, "language": language},
            config,
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessage):  # Filter to just model responses
                print(chunk.content, end="")
        print()
        

        

if __name__ == "__main__":
    chat_app = ChatApp()
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat...")
            break
        chat_app.add_message(user_input)
        # response = chat_app.get_response()
        chat_app.stream_response(chat_app.messages, "Italian", chat_app.config)

        # print(f"Bot: {response.content}")


