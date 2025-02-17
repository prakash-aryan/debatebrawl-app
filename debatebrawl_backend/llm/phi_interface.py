import requests
import json

OLLAMA_API_URL = "http://localhost:11434/api/generate"

def generate_phi(prompt: str, system: str = "") -> str:
    data = {
        "model": "phi3.5:3.8b-mini-instruct-q5_K_M",
        "prompt": prompt,
        "system": system,
        "stream": False
    }
    
    response = requests.post(OLLAMA_API_URL, json=data)
    if response.status_code == 200:
        return response.json()['response']
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

def get_debate_assistant_response(topic: str, user_argument: str, user_position: str) -> str:
    system = f"You are an AI debate assistant helping a user argue for the {user_position} position on the topic: '{topic}'. Provide constructive feedback and suggestions to improve their argument."
    prompt = f"User's argument: {user_argument}\n\nYour feedback and suggestions:"
    return generate_phi(prompt, system)