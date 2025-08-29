# Sentence Similarity Service

A high-performance FastAPI application for finding the most similar strings using the BAAI/bge-m3 sentence transformer model.

## Features

- REST API for text similarity matching
- Concurrent request handling
- Fast response times with model caching
- Docker support for easy deployment
- Health check endpoints

## Installation

### Using Python (local development)

1. Install the required packages:

```bash
pip install -r requirements_similarity.txt
```

2. Run the server:

```bash
python sentence_similarity_server.py
```

For production deployment, use the start script:

```bash
./start_similarity_server.sh
```

### Using Docker

1. Build the Docker image:

```bash
docker build -t similarity-service -f Dockerfile.similarity .
```

2. Run the container:

```bash
docker run -p 8080:8080 similarity-service
```

## API Usage

### Similarity Matching

**Endpoint:** `/similarity`

**Method:** POST

**Request Body:**

```json
{
  "query": ["chadnagar"],
  "options": ["gachibowli", "chandanagar", "himayatnagar"],
  "top_k": 2
}
```

**Response:**

```json
{
  "results": [
    {
      "chadnagar": [
        {
          "text": "chandanagar",
          "score": 0.9245
        },
        {
          "text": "himayatnagar",
          "score": 0.7856
        }
      ]
    }
  ],
  "time_taken": 0.3421
}
```

### Health Check

**Endpoint:** `/health`

**Method:** GET

## Testing

Run the test script to verify the API is working correctly:

```bash
python test_similarity_api.py
```

## Performance Tuning

- The server automatically scales based on available CPU cores
- Adjust the worker count in `start_similarity_server.sh` for your specific hardware
- For high-load scenarios, consider deploying behind a load balancer
