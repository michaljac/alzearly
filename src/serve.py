"""
FastAPI serving module for the ML project.
"""

from fastapi import FastAPI

# Initialize FastAPI app
app = FastAPI(
    title="ML Model Serving API",
    description="API for serving ML models",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "OK"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)