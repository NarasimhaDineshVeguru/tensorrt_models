FROM nvcr.io/nvidia/tritonserver:25.05-py3

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Copy model repository
COPY model_repository /model_repository

# Expose Triton ports
EXPOSE 8000 8001 8002

# Start Triton server
CMD ["tritonserver", "--model-repository=/model_repository"]
