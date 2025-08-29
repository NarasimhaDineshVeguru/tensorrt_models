import requests
import json
import time

def test_similarity_api():
    """Test the similarity API with a sample request"""
    url = "http://localhost:8080/similarity"
    
    # Test data
    payload = {
        "query": ["chadnagar"],
        "options": [
            "gachibowli",
            "chandanagar", 
            "himayatnagar"
        ],
        "top_k": 2
    }
    
    # Make request
    print("Sending request to similarity API...")
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload)
        elapsed = time.time() - start_time
        
        print(f"Response received in {elapsed:.2f} seconds")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\nResults:")
            print(json.dumps(result, indent=2))
            
            # Print in a more readable format
            for item in result["results"]:
                for query, matches in item.items():
                    print(f"\nQuery: {query}")
                    for match in matches:
                        print(f"  - {match['text']}: {match['score']:.4f}")
        else:
            print(f"Error: {response.text}")
    
    except Exception as e:
        print(f"Error testing API: {str(e)}")

if __name__ == "__main__":
    # First check if server is running
    try:
        health_response = requests.get("http://localhost:8080/health")
        if health_response.status_code == 200:
            print("Server is healthy")
            test_similarity_api()
        else:
            print(f"Server health check failed: {health_response.status_code}")
    except requests.exceptions.ConnectionError:
        print("Could not connect to server. Make sure it's running on http://localhost:8080")
