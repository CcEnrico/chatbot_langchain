# Use an official Python runtime as the base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app/src

# Expose the port (if different for each service, set this in docker-compose.yml)
EXPOSE 5001
EXPOSE 5002

# Default command (to be overridden in docker-compose.yml)
CMD ["python", "server.py"]
