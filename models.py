from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class LNAInput(BaseModel):
    freq_low: float
    freq_high: float
    noise_figure_db: Optional[float] = None
    gain_db: Optional[float] = None

class BiasTInput(BaseModel):
    freq_low: float
    freq_high: float

class ExtractRequest(BaseModel):
    prompt: str

class ExtractResponse(BaseModel):
    freq_low: Optional[float] = None
    freq_high: Optional[float] = None
    gain_db: Optional[float] = None
    noise_figure_db: Optional[float] = None

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = datetime.now()

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    recommendations: Optional[dict] = None

class LNARecommendation(BaseModel):
    part_number: str
    manufacturer: str
    frequency_range: str
    gain: str
    noise_figure: str
    package: str
    datasheet_url: str

class BiasTRecommendation(BaseModel):
    part_number: str
    manufacturer: str
    frequency_range: str
    insertion_loss: str
    return_loss: str
    max_dc_voltage: str
    max_dc_current: str
    connector_type: str
    datasheet_url: str

class RecommendationResponse(BaseModel):
    recommendations: List[LNARecommendation]

class BiasTRecommendationResponse(BaseModel):
    recommendations: List[BiasTRecommendation]