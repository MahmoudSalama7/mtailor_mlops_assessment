# Dockerfile for Cerebrium Deployment

# Use an official Python runtime as a parent image
# Choose a version compatible with your dependencies (e.g., 3.9, 3.10, 3.11)
# Consider using slim variants if size is critical, but ensure all system deps are met.
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies if needed (e.g., for OpenCV or other libraries)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libgl1-mesa-glx libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt ./

# Install Python dependencies
# Using --no-cache-dir can reduce image size
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# Copy the rest of the application code into the container
# This includes model.py, main.py, the ONNX model, etc.
COPY . . 

# Ensure the ONNX model file is present (it should be copied by `COPY . .`)
# You might need to run convert_to_onnx.py *before* building the Docker image
# or include steps here to generate it if needed, though that increases build time.
RUN if [ ! -f model.onnx ]; then \
    echo "Error: model.onnx not found. Please generate it before building the Docker image."; \
    exit 1; \
    fi

# Define the command to run the application
# Cerebrium typically looks for a FastAPI app instance named `app` in `main.py`
# The command might be slightly different based on Cerebrium's specific requirements
# for custom Docker deployments. Refer to Cerebrium documentation.
# Usually, Cerebrium handles the CMD/ENTRYPOINT, but specifying it can be useful for local testing.
# Example for running FastAPI with Uvicorn (adjust host/port as needed for Cerebrium):
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

# Note: Cerebrium's environment might inject specific environment variables or expect
# the application to start in a particular way. Consult their documentation for
# custom Docker image deployment guidelines.

