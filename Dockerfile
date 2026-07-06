FROM python:3.11-slim

WORKDIR /app

# Install CPU-only PyTorch first to dramatically reduce build time and download size (~150MB vs ~800MB CUDA wheel)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements from python_backend subdirectory
COPY apps/python_backend/requirements.txt .

# Install remaining dependencies (pip will skip installing full PyTorch as it's already satisfied)
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code from the python_backend subdirectory to /app
COPY apps/python_backend/ .

# Expose port 8000
EXPOSE 8000

# Start uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
