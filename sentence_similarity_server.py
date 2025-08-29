import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
import numpy as np
import asyncio
import logging
from contextlib import asynccontextmanager
import time
import os
from functools import lru_cache
# kill $(lsof -t -i:5003)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Model cache with LRU decorator
@lru_cache(maxsize=1)
def get_model():
    logger.info("Loading sentence transformer model...")
    start_time = time.time()
    try:
        # Try to load the model on GPU
        model = SentenceTransformer("BAAI/bge-m3", device="cuda")
        logger.info("Model loaded on GPU")
    except Exception as e:
        logger.warning(f"Could not load model on GPU: {e}. Falling back to CPU.")
        model = SentenceTransformer("BAAI/bge-m3", device="cpu")
        logger.info("Model loaded on CPU")
    
    logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
    return model

# Application startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    logger.info("Starting application, preloading model...")
    get_model()  # Preload the model
    yield
    # Cleanup on shutdown
    logger.info("Shutting down application...")

# Initialize FastAPI app
app = FastAPI(
    title="Sentence Similarity API",
    description="API for finding the most similar strings using sentence transformers",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input models
class SimilarityRequest(BaseModel):
    query: List[str] | str = Field(..., description="List of query strings to compare", example=["chadnagar"])
    options: List[Dict[str, str]] = Field(..., description="List of options to match against", 
                               example=[{"label": "gachibowli"}, {"label": "chandanagar"}, {"label": "himayatnagar"}])
    top_k: Optional[int] = Field(1, description="Number of top matches to return per query")

class SimilarityMatch(BaseModel):
    text: str
    score: float

class SimilarityMatch(BaseModel):
    text: str
    score: float

class LabelValueMatch(BaseModel):
    label: str
    value: str

class SimilarityResponse(BaseModel):
    results: Union[List[Dict[str, Any]], List[LabelValueMatch], List] = Field(
        default=[],
        description="Results in either [] format or [{label, value}] format"
    )
    time_taken: float = 0.0

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the service is running"""
    return {"status": "healthy"}

@app.post("/similarity", response_model=SimilarityResponse)
async def get_similarity(request:SimilarityRequest):
    print("request", request)
    """
    Find the most similar options for each query string
    """
    start_time = time.time()
    
    if not request.query:
        raise HTTPException(status_code=400, detail="Query list cannot be empty")
    
    if not request.options:
        raise HTTPException(status_code=400, detail="Options list cannot be empty")
    
    # Ensure top_k is valid
    top_k = min(request.top_k, len(request.options))
    if top_k < 1:
        top_k = 1
    
    try:
        model = get_model()

        options = request.options

        compare_options = [i["label"] for i in options]
        
        # Create embeddings
        query_embeddings = model.encode(request.query, convert_to_tensor=True, normalize_embeddings=True)
        option_embeddings = model.encode(compare_options, convert_to_tensor=True, normalize_embeddings=True)
        
        # Calculate similarities
        similarities = model.similarity(query_embeddings, option_embeddings)
        print("similarities", similarities)
        max_index = similarities.argmax()
        print("max_index", max_index)
        if similarities[0][max_index] < 0.5:
            return SimilarityResponse(
                results=[{}],
                time_taken=time.time() - start_time
            )
        else:
            return SimilarityResponse(
                results=[{
                    "label": options[max_index]["label"],
                    "value": options[max_index].get("value", options[max_index]["label"])
                }],
                time_taken=time.time() - start_time
            )
        # Process results
        # results = []
        # for idx_i, query in enumerate(request.query):
        #     query_similarities = similarities[idx_i]
            
        #     # Get indices of top-k similar options
        #     top_indices = np.argsort(query_similarities)[::-1][:top_k]
            
        #     # Construct matches
        #     matches = [
        #         SimilarityMatch(
        #             text=request.options[idx_j],
        #             score=float(query_similarities[idx_j])
        #         )
        #         for idx_j in top_indices
        #     ]
            
        #     results.append({query: matches})
        
        # time_taken = time.time() - start_time
        # logger.info(f"Processed similarity request in {time_taken:.4f} seconds")
        
        # return SimilarityResponse(
        #     results=results,
        #     time_taken=time_taken
        # )
    
    except Exception as e:
        logger.error(f"Error processing similarity request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with usage information"""
    return {
        "message": "Sentence Similarity API is running",
        "usage": "POST to /similarity with query and options",
        "example": {
            "query": ["chadnagar"],
            "options": ["gachibowli", "chandanagar", "himayatnagar"],
            "top_k": 2
        }
    }

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 5003))
    
    # Run the server
    logger.info(f"Starting server on port {port}")
    uvicorn.run("sentence_similarity_server:app", 
                host="0.0.0.0", 
                port=port,
                workers=1,  # Adjust based on available CPU cores
                log_level="info")
