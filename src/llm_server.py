from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=10000, temperature=0.9 )


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in Italian.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

@app.route('/invoke', methods=['POST'])
def invoke():
    messages = request.json['messages']
    response = llm.invoke(messages)
    return jsonify({"response": response.content})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)