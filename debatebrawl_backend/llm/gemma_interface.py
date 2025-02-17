import requests
import json
from typing import List

OLLAMA_API_URL = "http://localhost:11434/api/generate"

def generate_gemma(prompt: str, system: str = "") -> str:
    data = {
        "model": "gemma2:2b-instruct-q8_0",
        "prompt": prompt,
        "system": system,
        "stream": False
    }
    
    response = requests.post(OLLAMA_API_URL, json=data)
    if response.status_code == 200:
        return response.json()['response']
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

def get_ai_opponent_response(topic: str, user_argument: str, ai_position: str) -> str:
    system = f"You are an AI debater taking the {ai_position} position on the topic: '{topic}'. Respond to the user's argument with a strong counter-argument."
    prompt = f"User's argument: {user_argument}\n\nYour response:"
    return generate_gemma(prompt, system)

def generate_llm_suggestions(topic: str, position: str) -> List[str]:
    system = f"You are an AI debate assistant. Generate 3 strong argument suggestions for the {position} position on the topic: '{topic}'."
    prompt = "Generate 3 debate argument suggestions."
    response = generate_gemma(prompt, system)
    return [suggestion.strip() for suggestion in response.split('\n') if suggestion.strip()]