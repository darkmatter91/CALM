FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libproj-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a script that will run the server
RUN echo '#!/bin/bash\n\
python app.py\n\
' > /app/docker-entrypoint.sh

# Make the script executable
RUN chmod +x /app/docker-entrypoint.sh

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_DEBUG=0

# Expose port
EXPOSE 5000

# Run the application
ENTRYPOINT ["/app/docker-entrypoint.sh"] 