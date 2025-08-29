import requests
import json
import numpy as np

# Test inference with the Triton server
def test_inference():
    url = "http://localhost:8000/v2/models/intent_classification/infer"
    
    # Sample text for classification
    test_text = "I want to book a flight to New York"
    
    # Prepare the inference request
    data = {
        "inputs": [
            {
                "name": "text_input",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [test_text]
            }
        ]
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Predicted Label: {result['outputs'][0]['data'][0]}")
            print(f"Confidence Score: {result['outputs'][1]['data'][0]}")
        else:
            print("Error in inference")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # First check if server is alive
    try:
        health_response = requests.get("http://localhost:8000/v2/health/live")
        print(f"Server health: {health_response.status_code}")
    except Exception as e:
        print(f"Server not reachable: {e}")
        exit(1)
    
    # Check model status
    try:
        model_response = requests.get("http://localhost:8000/v2/models/intent_classification")
        print(f"Model status: {model_response.status_code}")
        if model_response.status_code == 200:
            print("Model is ready for inference")
        else:
            print(f"Model not ready: {model_response.text}")
    except Exception as e:
        print(f"Error checking model: {e}")
    
    # Run inference test
    test_inference()
