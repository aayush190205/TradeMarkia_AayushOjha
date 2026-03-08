from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cache import SemanticCache

# Justification: Initializing the FastAPI app and the Semantic Cache at the module level
# ensures the heavy machine learning models (GMM, SentenceTransformer) and the in-memory 
# vector database are loaded into memory exactly once when the server starts.
app = FastAPI(title="Trademarkia Semantic Search & Cache API")
semantic_cache = SemanticCache(threshold=0.75)

# Pydantic model to strictly enforce the expected JSON body for the POST request
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def handle_query(request: QueryRequest):
    """
    Accepts a natural language query, checks the cluster-routed semantic cache, 
    and returns the best matching document.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    # The heavy lifting (embedding, clustering, and caching logic) is abstracted away 
    # in the SemanticCache class to keep the API layer clean and strictly concerned with HTTP.
    response = semantic_cache.process_query(request.query)
    
    return response

@app.get("/cache/stats")
async def handle_get_stats():
    """
    Returns the current state and performance metrics of the semantic cache.
    """
    return semantic_cache.get_stats()

@app.delete("/cache")
async def handle_clear_cache():
    """
    Flushes the cache entirely and resets all performance statistics.
    """
    semantic_cache.clear_cache()
    return {"message": "Cache successfully flushed and statistics reset."}