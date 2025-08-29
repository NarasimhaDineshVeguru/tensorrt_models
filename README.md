# Triton Inference Server Setup Guide

This repository contains models and configuration for running NVIDIA's Triton Inference Server with custom Python models. This guide will walk you through setting up and running the Triton server from scratch.

## Overview

This repository contains:
- PyTorch/Hugging Face transformer models deployed as Python backend models in Triton
- A FastAPI sentence similarity service
- Docker configurations for easy deployment

## Prerequisites

- NVIDIA GPU (recommended)
- NVIDIA Docker (for Docker-based deployment)
- Python 3.10+
- pip package manager

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/NarasimhaDineshVeguru/tensorrt_models.git
cd tensorrt_models
```

### 2. Set Up Python Environment

Create and activate a virtual environment:

```bash
python -m venv env
source env/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

For the sentence similarity service, install additional dependencies:

```bash
pip install -r requirements_similarity.txt
```

### 3. Understanding Triton Model Repository Structure

The Triton Inference Server requires a specific directory structure for models:

```
model_repository/
  ├── model_name/
  │   ├── config.pbtxt       # Model configuration
  │   └── 1/                 # Model version
  │       └── model.py       # Python model implementation
```

Each model folder must contain:
- `config.pbtxt`: Defines model inputs/outputs and runtime configurations
- Version directories (numbered, like `1/`): Contains the actual model files

### 4. Model Configuration

Each model's `config.pbtxt` specifies:
- Input and output tensor specifications
- Execution resources (CPU/GPU)
- Batch size configuration

Example from the intent model:

```
name: "intent"
backend: "python"
max_batch_size: 8
input [
  {
    name: "text_input"
    data_type: TYPE_STRING
    dims: [1]
  }
]
output [
  {
    name: "label"
    data_type: TYPE_STRING
    dims: [1]
  },
  {
    name: "score"
    data_type: TYPE_FP32
    dims: [1]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

### 5. Python Backend Model Implementation

For Python backend models, the `model.py` file must implement the `TritonPythonModel` class with specific methods:

```python
import json
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import pipeline

class TritonPythonModel:
    def initialize(self, args):
        """Initialize the model and load resources."""
        self.model_config = json.loads(args['model_config'])
        # Load model here
        self.classifier = pipeline(
            "text-classification", 
            model="model_name_or_path",
            return_all_scores=True
        )

    def execute(self, requests):
        """Process inference requests."""
        responses = []
        for request in requests:
            # Get input
            input_tensor = pb_utils.get_input_tensor_by_name(request, "text_input")
            input_text = input_tensor.as_numpy()
            
            # Process input
            result = self.classifier(input_text[0][0].decode('utf-8'))
            
            # Create output tensors
            output0 = pb_utils.Tensor("label", np.array([result[0]['label']], dtype=np.object_))
            output1 = pb_utils.Tensor("score", np.array([result[0]['score']], dtype=np.float32))
            
            # Create and append inference response
            inference_response = pb_utils.InferenceResponse(output_tensors=[output0, output1])
            responses.append(inference_response)
        
        return responses

    def finalize(self):
        """Clean up resources."""
        pass
```

### 6. Running Triton Server

#### Using Docker (Recommended)

1. Build the Docker image:

```bash
sudo docker build -t triton-intent-classifier .
```

2. Run the Docker container:

```bash
sudo docker run --gpus all --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/model_repository:/model_repository triton-intent-classifier tritonserver --model-repository=/model_repository
```

This will start Triton server with the following endpoints:
- HTTP endpoint: `localhost:8000`
- gRPC endpoint: `localhost:8001`
- Metrics endpoint: `localhost:8002`

#### Running Locally (Without Docker)

If you have Triton installed locally:

```bash
tritonserver --model-repository=/path/to/model_repository
```

### 7. Testing Inference

You can test the server using the provided `test_inference.py` script:

```bash
python test_inference.py
```

Or use curl for HTTP requests:

```bash
curl -X POST http://localhost:8000/v2/models/intent/infer -d '{
  "inputs": [
    {
      "name": "text_input",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": ["I want to book a flight"]
    }
  ]
}'
```

### 8. Running the Sentence Similarity Service

For the sentence similarity service:

```bash
python sentence_similarity_server.py
```

Or for production:

```bash
./start_similarity_server.sh
```

Using Docker:

```bash
docker build -t similarity-service -f Dockerfile.similarity .
docker run -p 8080:8080 similarity-service
```

## Advanced Configuration

### GPU Configuration

To specify which GPUs to use:

```bash
docker run --gpus '"device=0,1"' -p 8000:8000 -p 8001:8001 -p 8002:8002 triton-server
```

### Memory Optimization

For large models, adjust the model configuration in `config.pbtxt`:

```
optimization {
  execution_accelerators {
    gpu_execution_accelerator {
      name: "tensorrt"
    }
  }
  cuda {
    graphs: true
  }
}
```

### Dynamic Batching

Enable dynamic batching for better throughput:

```
dynamic_batching {
  preferred_batch_size: [4, 8]
  max_queue_delay_microseconds: 5000
}
```

## Monitoring

Access Triton metrics at:
- http://localhost:8002/metrics

## Troubleshooting

Common issues:

1. **CUDA Out of Memory**: Reduce batch size or model size
2. **Model Loading Failures**: Check paths and dependencies
3. **Slow Inference**: Enable dynamic batching or TensorRT optimization

For more detailed logs, run Triton with increased verbosity:

```bash
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 triton-server tritonserver --model-repository=/model_repository --log-verbose=1
```

## References

- [NVIDIA Triton Inference Server Documentation](https://github.com/triton-inference-server/server)
- [Triton Python Backend Documentation](https://github.com/triton-inference-server/python_backend)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
