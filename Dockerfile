FROM python:3.11-slim

WORKDIR /app

# Copy requirements from python_backend subdirectory
COPY apps/python_backend/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code from the python_backend subdirectory to /app
COPY apps/python_backend/ .

# Expose port 8000
EXPOSE 8000

# Start uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
