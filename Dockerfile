# Use a compatible Python version
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code
COPY src/ ./src/
COPY database/ ./database/

# Expose the port for the server
EXPOSE 5001

# Run the server
CMD ["python", "src/RAG_server.py"]
