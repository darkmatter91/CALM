FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
# Don't create .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Don't buffer stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
# Set production by default
ENV FLASK_ENV=production

# Install system dependencies required for libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgeos-dev \
    libproj-dev \
    proj-bin \
    libgdal-dev \
    build-essential \
    curl \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (leveraging cache layers)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create volume for persistent data
VOLUME /app/data

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/ || exit 1

# Command to run the application
CMD ["python", "app.py"] 