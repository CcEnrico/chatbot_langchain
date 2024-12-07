from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Flask app setup
app = Flask(__name__)

# Language model setup
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000, temperature=0.5, frequency_penalty=0.2)


# Prompt template with detailed response instruction
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Provide thorough, detailed, and elaborate answers to all questions in Italian.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

@app.route('/invoke', methods=['POST'])
def invoke():
    # Extract user messages from the request
    messages = request.json.get('messages', [])
    
    # Check if messages are provided
    if not messages:
        return jsonify({"error": "No messages provided"}), 400
    
    # Get response from the language model
    response = llm.invoke(messages)
    
    # Return response as JSON
    return jsonify({"response": response.content})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
