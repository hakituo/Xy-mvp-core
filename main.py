import sys
import os
# Ensure current directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from fastapi import FastAPI, WebSocket
from shared.di import container
from domain.interfaces.base_interfaces import LLMInterface, MemoryInterface
from data.adapters.local_llm_adapter import LocalLLMAdapter
from data.adapters.mock_llm_adapter import MockLLMAdapter
from data.repositories.memory_repository import InMemoryMemoryRepository
from domain.entities.character import CharacterProfile
from domain.services.chat_service import ChatService
from presentation.websocket.handler import WebSocketHandler
from config import get_settings
import logging

# Setup Logger
logging.basicConfig(level=logging.INFO)

# Bootstrap
def bootstrap():
    # Register Dependencies
    # Check if model exists
    settings = get_settings()
    if os.path.exists(settings.model.text_path):
        logging.info(f"Using LocalLLMAdapter with model at {settings.model.text_path}")
        container.register(LLMInterface, LocalLLMAdapter()) 
    else:
        logging.warning(f"Model not found at {settings.model.text_path}, using MockLLMAdapter")
        container.register(LLMInterface, MockLLMAdapter())
        
    container.register(MemoryInterface, InMemoryMemoryRepository())

    # Initialize Character (Mock Aveline for MVP)
    aveline = CharacterProfile(
        name="Aveline",
        system_prompt="You are Aveline, a digital assistant with emotions.",
        sensory_triggers=[], # Add triggers as needed
        behavior_chains=[]
    )
    
    return aveline

app = FastAPI()
character = bootstrap()
chat_service = ChatService(character)
ws_handler = WebSocketHandler(chat_service)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_handler.handle_connection(websocket)

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "mvp-core-1.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
