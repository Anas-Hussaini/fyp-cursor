import os
import json
import uuid
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from models import ChatMessage, ChatRequest, ChatResponse
from nlp_extractor import extract_requirements_via_openai
from lna_model import predict_lna
from bias_t_model import recommend_bias_t

class ConversationManager:
    def __init__(self):
        self.conversations: Dict[str, List[ChatMessage]] = {}
        
    def _get_openai_client(self):
        """Get OpenAI client with error handling"""
        # Load API key dynamically
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            print("OpenAI API key not found or invalid")
            return None
            
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            # Test the client with a simple call
            return client
        except Exception as e:
            print(f"Error creating OpenAI client: {e}")
            return None
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for the RF component chatbot"""
        return """You are an expert RF component recommendation assistant. You help users find the perfect LNA (Low Noise Amplifier) and Bias-T components for their RF systems.

Your capabilities:
1. Understand RF requirements from natural language
2. Recommend appropriate LNA and Bias-T components
3. Explain technical specifications
4. Provide helpful guidance on RF system design

When users ask for component recommendations:
1. Extract their requirements (frequency range, gain, noise figure, etc.)
2. Provide specific component recommendations with part numbers
3. Explain why these components are suitable
4. Offer additional advice if needed

Be conversational, helpful, and technically accurate. If you don't have enough information, ask clarifying questions."""
    
    def _format_recommendations(self, lna_result: dict, bias_t_result: dict) -> str:
        """Format component recommendations for chat response"""
        if not lna_result or not bias_t_result:
            return "I couldn't find suitable components with the given specifications. Please try different parameters."
            
        response = "\n\n**Component Recommendations:**\n\n"
        
        # LNA Recommendation
        response += "**📡 LNA (Low Noise Amplifier):**\n"
        response += f"• Part Number: {lna_result.get('part_number', 'N/A')}\n"
        response += f"• Manufacturer: {lna_result.get('manufacturer', 'N/A')}\n"
        response += f"• Frequency Range: {lna_result.get('frequency_range', 'N/A')}\n"
        response += f"• Gain: {lna_result.get('gain', 'N/A')}\n"
        response += f"• Noise Figure: {lna_result.get('noise_figure', 'N/A')}\n"
        response += f"• Package: {lna_result.get('package', 'N/A')}\n"
        
        # Bias-T Recommendation
        response += "\n**⚡ Bias-T:**\n"
        response += f"• Part Number: {bias_t_result.get('part_number', 'N/A')}\n"
        response += f"• Manufacturer: {bias_t_result.get('manufacturer', 'N/A')}\n"
        response += f"• Frequency Range: {bias_t_result.get('frequency_range', 'N/A')}\n"
        response += f"• Insertion Loss: {bias_t_result.get('insertion_loss', 'N/A')}\n"
        response += f"• Max DC Voltage: {bias_t_result.get('max_dc_voltage', 'N/A')}\n"
        response += f"• Max DC Current: {bias_t_result.get('max_dc_current', 'N/A')}\n"
        
        response += "\nThese components should work well together for your RF system requirements."
        return response
    
    def _get_component_recommendations(self, extracted_data: dict) -> Tuple[Optional[dict], Optional[dict]]:
        """Get component recommendations based on extracted data"""
        try:
            # Prepare LNA input
            lna_input = {
                'freq_low': extracted_data.get('freq_low', 0),
                'freq_high': extracted_data.get('freq_high', 0),
                'gain_db': extracted_data.get('gain_db'),
                'noise_figure_db': extracted_data.get('noise_figure_db')
            }
            
            # Prepare Bias-T input
            bias_t_input = {
                'freq_low': extracted_data.get('freq_low', 0),
                'freq_high': extracted_data.get('freq_high', 0)
            }
            
            # Get recommendations
            lna_result = predict_lna(lna_input)
            bias_t_results = recommend_bias_t(bias_t_input, top_k=1)
            bias_t_result = bias_t_results[0] if bias_t_results else None
            
            return lna_result, bias_t_result
            
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return None, None
    
    def process_message(self, request: ChatRequest) -> ChatResponse:
        """Process a chat message and return response"""
        # Get or create conversation
        conversation_id = request.conversation_id or str(uuid.uuid4())
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        # Add user message to conversation
        user_message = ChatMessage(role="user", content=request.message)
        self.conversations[conversation_id].append(user_message)
        
        # Get OpenAI client
        client = self._get_openai_client()
        
        if not client:
            # Fallback response without OpenAI
            response = "I'm currently unable to access advanced AI features. Please try again later or contact support."
            assistant_message = ChatMessage(role="assistant", content=response)
            self.conversations[conversation_id].append(assistant_message)
            return ChatResponse(response=response, conversation_id=conversation_id)
        
        try:
            # Prepare conversation history for OpenAI
            messages = [{"role": "system", "content": self._create_system_prompt()}]
            
            # Add conversation history (last 10 messages to avoid token limits)
            recent_messages = self.conversations[conversation_id][-10:]
            for msg in recent_messages:
                messages.append({"role": msg.role, "content": msg.content})
            
            # Get OpenAI response
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            assistant_response = response.choices[0].message.content
            
            # Check if user is asking for component recommendations
            if any(keyword in request.message.lower() for keyword in ['recommend', 'lna', 'bias-t', 'bias t', 'amplifier', 'component']):
                # Extract requirements
                extracted_data = extract_requirements_via_openai(request.message)
                
                if extracted_data.get('freq_low') and extracted_data.get('freq_high'):
                    # Get component recommendations
                    lna_result, bias_t_result = self._get_component_recommendations(extracted_data)
                    
                    if lna_result and bias_t_result:
                        # Add recommendations to response
                        recommendations_text = self._format_recommendations(lna_result, bias_t_result)
                        assistant_response += recommendations_text
                        
                        # Store recommendations data
                        recommendations_data = {
                            'lna': lna_result,
                            'bias_t': bias_t_result
                        }
                    else:
                        assistant_response += "\n\nI couldn't find suitable components with the given specifications. Please try different parameters."
                        recommendations_data = None
                else:
                    assistant_response += "\n\nI need more specific information about your requirements. Please specify frequency range, gain, noise figure, etc."
                    recommendations_data = None
            else:
                recommendations_data = None
            
            # Add assistant message to conversation
            assistant_message = ChatMessage(role="assistant", content=assistant_response)
            self.conversations[conversation_id].append(assistant_message)
            
            return ChatResponse(
                response=assistant_response,
                conversation_id=conversation_id,
                recommendations=recommendations_data
            )
            
        except Exception as e:
            error_response = f"I encountered an error: {str(e)}. Please try again."
            assistant_message = ChatMessage(role="assistant", content=error_response)
            self.conversations[conversation_id].append(assistant_message)
            return ChatResponse(response=error_response, conversation_id=conversation_id)
    
    def get_conversation_history(self, conversation_id: str) -> List[ChatMessage]:
        """Get conversation history"""
        return self.conversations.get(conversation_id, [])
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear a conversation"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            return True
        return False

# Global conversation manager instance
conversation_manager = ConversationManager()
