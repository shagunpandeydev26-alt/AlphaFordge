# API for serving the trained RL model, enabling external access via HTTP requests.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path

from .inference import ModelInferenceAPI, TradingInferenceEngine
from ..utils.logger import setup_logger

# Setup logger
logger = setup_logger(__name__)

# FastAPI app
app = FastAPI(title="RL Trading Agent API", version="1.0.0")

# Initialize inference API
models_dir = Path(__file__).parent.parent.parent / "models"
inference_api = ModelInferenceAPI(models_dir)

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    ticker: str
    portfolio_value: float
    num_shares: int

class BatchPredictionRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    initial_amount: float = 10000.0

class PredictionResponse(BaseModel):
    ticker: str
    action: int
    recommendation: str
    confidence: Optional[float] = None

class BatchPredictionResponse(BaseModel):
    ticker: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    final_value: float
    metrics: Dict

class ModelStatusResponse(BaseModel):
    available_models: List[str]
    total_models: int

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "RL Trading Agent API is running"}

@app.get("/models", response_model=ModelStatusResponse)
async def get_available_models():
    """Get list of available trained models."""
    try:
        model_files = list(models_dir.glob("*.zip"))
        tickers = [f.stem for f in model_files]
        
        return ModelStatusResponse(
            available_models=tickers,
            total_models=len(tickers)
        )
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/predict", response_model=PredictionResponse)
async def predict_action(request: PredictionRequest):
    """Get trading recommendation for a specific ticker."""
    try:
        logger.info(f"Prediction request for {request.ticker}")
        
        action = inference_api.predict_for_ticker(
            request.ticker,
            request.portfolio_value,
            request.num_shares
        )
        
        if action is None:
            raise HTTPException(
                status_code=404, 
                detail=f"Model not found for ticker {request.ticker}"
            )
        
        # Convert action to recommendation
        if action > 0:
            recommendation = "BUY"
        elif action < 0:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"
        
        return PredictionResponse(
            ticker=request.ticker,
            action=action,
            recommendation=recommendation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_prediction(request: BatchPredictionRequest):
    """Run batch prediction over a date range."""
    try:
        logger.info(f"Batch prediction request for {request.ticker}")
        
        model = inference_api.load_model(request.ticker)
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found for ticker {request.ticker}"
            )
        
        results = inference_api.inference_engine.batch_predict(
            model,
            request.ticker,
            request.start_date,
            request.end_date,
            request.initial_amount
        )
        
        return BatchPredictionResponse(
            ticker=request.ticker,
            total_return=results['metrics']['total_return'],
            sharpe_ratio=results['metrics']['sharpe_ratio'],
            max_drawdown=results['metrics']['max_drawdown'],
            final_value=results['final_value'],
            metrics=results['metrics']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")

@app.post("/evaluate/{ticker}")
async def evaluate_model(ticker: str, test_start: str, test_end: str):
    """Evaluate model performance on test data."""
    try:
        logger.info(f"Model evaluation request for {ticker}")
        
        model = inference_api.load_model(ticker)
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found for ticker {ticker}"
            )
        
        metrics = inference_api.inference_engine.evaluate_model(
            model, ticker, test_start, test_end
        )
        
        return {
            "ticker": ticker,
            "evaluation_period": f"{test_start} to {test_end}",
            "metrics": metrics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        raise HTTPException(status_code=500, detail="Model evaluation failed")

@app.get("/health")
async def health_check():
    """Detailed health check."""
    try:
        # Check if models directory exists
        models_exist = models_dir.exists()
        model_count = len(list(models_dir.glob("*.zip"))) if models_exist else 0
        
        return {
            "status": "healthy",
            "models_directory_exists": models_exist,
            "available_models": model_count,
            "api_version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
