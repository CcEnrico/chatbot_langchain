from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import trim_messages
from langchain_openai import ChatOpenAI

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=500, temperature=0.6)

# Configure the trimmer
trimmer = trim_messages(
    max_tokens=50,
    strategy="last",
    token_counter=llm,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# Test messages
test_messages = [
    SystemMessage(content="System initialization message."),
    HumanMessage(content="Hi, I need help with a problem."),
    AIMessage(content="Sure! What problem do you need help with?"),
    HumanMessage(content="I'm trying to figure out why my code isn't working."),
    AIMessage(content="Can you share the specific error or unexpected behavior?"),
    HumanMessage(content="Yes, I'm seeing an index out of range error."),
]

def main():
    print("Original messages:")
    for msg in test_messages:
        print(f"[{msg.__class__.__name__}] {msg.content}")

    try:
        print("\nTrimmed messages:")
        trimmed_messages = trimmer.invoke(test_messages)
        for msg in trimmed_messages:
            print(f"[{msg.__class__.__name__}] {msg.content}")
    except Exception as e:
        print(f"Error during trimming: {e}")

if __name__ == "__main__":
    main()
