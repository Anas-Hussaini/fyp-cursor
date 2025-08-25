from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from models import (
    LNAInput,
    BiasTInput,
    RecommendationResponse,
    BiasTRecommendationResponse,
    LNARecommendation,
    BiasTRecommendation,
    ExtractRequest,
    ExtractResponse,
    ChatRequest,
    ChatResponse,
)
from lna_model import predict_lna
from bias_t_model import recommend_bias_t
from chat_manager import conversation_manager
import os
import uvicorn
from dotenv import load_dotenv
from nlp_extractor import extract_requirements_via_openai

# Load environment variables
load_dotenv()

app = FastAPI(
    title="RF Component Recommendation API",
    description="API for recommending LNA and Bias-T components based on specifications",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for the web interface
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return {
        "message": "RF Component Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "lna_recommendation": "/recommend-lna",
            "bias_t_recommendation": "/recommend-bias-t",
            "extract_requirements": "/extract-requirements",
            "web_interface": "/interface",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}

@app.get("/interface")
async def web_interface():
    """Serve the web interface"""
    return FileResponse("static/index.html")

@app.post("/extract-requirements", response_model=ExtractResponse)
async def extract_requirements(req: ExtractRequest):
    try:
        data = extract_requirements_via_openai(req.prompt)
        return ExtractResponse(**data)
    except Exception as e:
        # Fail soft with empty extraction
        return ExtractResponse()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint for conversational AI"""
    try:
        response = conversation_manager.process_message(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/conversation/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """Get conversation history"""
    try:
        history = conversation_manager.get_conversation_history(conversation_id)
        return {"conversation_id": conversation_id, "messages": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation: {str(e)}")

@app.delete("/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Clear a conversation"""
    try:
        success = conversation_manager.clear_conversation(conversation_id)
        if success:
            return {"message": "Conversation cleared successfully"}
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing conversation: {str(e)}")

@app.post("/recommend-lna", response_model=RecommendationResponse)
async def recommend_lna(input_data: LNAInput):
    """Recommend LNA components based on input specifications"""
    try:
        # Convert input data to dictionary
        input_features = input_data.model_dump(exclude_none=True)
        
        # Get recommendation
        result = predict_lna(input_features)
        
        # Create LNA recommendation object
        lna_rec = LNARecommendation(
            part_number=result['part_number'],
            manufacturer=result['manufacturer'],
            frequency_range=result['frequency_range'],
            gain=result['gain'],
            noise_figure=result['noise_figure'],
            package=result['package'],
            datasheet_url=result['datasheet_url']
        )
        
        return RecommendationResponse(recommendations=[lna_rec])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during LNA recommendation: {str(e)}")

@app.post("/recommend-bias-t", response_model=BiasTRecommendationResponse)
async def recommend_bias_t_endpoint(input_data: BiasTInput):
    """Recommend Bias-T components based on input specifications"""
    try:
        # Convert input data to dictionary
        input_features = input_data.model_dump(exclude_none=True)
        
        # Get recommendations
        results = recommend_bias_t(input_features, top_k=1)
        
        # Convert to BiasTRecommendation objects
        bias_t_recs = []
        for result in results:
            bias_t_rec = BiasTRecommendation(
                part_number=result['Part Number'],
                manufacturer=result['Manufacturer'],
                frequency_range=result['Frequency Range'],
                insertion_loss=result['Insertion Loss'],
                return_loss=result['Return Loss'],
                max_dc_voltage=result['Max DC Voltage'],
                max_dc_current=result['Max DC Current'],
                connector_type=result['Connector Type'],
                datasheet_url=result['Datasheet URL']
            )
            bias_t_recs.append(bias_t_rec)
        
        return BiasTRecommendationResponse(recommendations=bias_t_recs)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during Bias-T recommendation: {str(e)}")

# For Vercel deployment
if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Starting RF Component Recommendation API")
    print(f"📡 Server will be available at:")
    print(f"   - Local: http://localhost:{port}")
    print(f"   - Network: http://{host}:{port}")
    print(f"   - API Documentation: http://localhost:{port}/docs")
    print(f"   - Web Interface: http://localhost:{port}/interface")
    print(f"   - Health Check: http://localhost:{port}/health")
    print(f"\nPress Ctrl+C to stop the server")
    
    uvicorn.run(app, host=host, port=port) 