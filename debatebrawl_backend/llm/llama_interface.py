import requests
import json
import re
from typing import List

OLLAMA_API_URL = "http://localhost:11434/api/generate"

def generate_llama(prompt: str, system: str = "") -> str:
    data = {
        "model": "llama3.2:3b-instruct-q8_0",
        "prompt": prompt,
        "system": system,
        "stream": False
    }
    
    response = requests.post(OLLAMA_API_URL, json=data)
    if response.status_code == 200:
        return response.json()['response']
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

def generate_debate_topics() -> List[str]:
    system = "You are a debate topic generator. Generate 5 concise, trending, and controversial debate topics suitable for a 10-minute debate session. Provide only the topics, without any explanations or numbering."
    prompt = "Generate 5 debate topics."
    response = generate_llama(prompt, system)
    topics = [topic.strip() for topic in response.split('\n') if topic.strip()]
    return clean_topics(topics)

def clean_topics(topics: List[str]) -> List[str]:
    cleaned_topics = []
    for topic in topics:
        # Remove any numbering or prefixes
        topic = re.sub(r'^\d+\.?\s*', '', topic.strip())
        # Remove any explanatory text after the main topic
        topic = re.split(r'[.:]', topic)[0].strip()
        if topic and not topic.lower().startswith(("here are", "certainly", "sure,", "the following", "these are")):
            cleaned_topics.append(topic)
    return cleaned_topics[:5]  # Ensure we return at most 5 topics

def generate_argument(topic: str, stance: str, context: str = "", num_suggestions: int = 1, strategy: str = "") -> List[str]:
    system = f"You are an AI debate assistant. Generate {num_suggestions} strong argument(s) {stance} the motion for the debate topic: '{topic}'. Context: {context}. Strategy: {strategy}"
    prompt = f"Generate {num_suggestions} debate argument(s)."
    response = generate_llama(prompt, system)
    return [arg.strip() for arg in response.split('\n') if arg.strip()]

def evaluate_argument(argument: str, topic: str) -> str:
    system = f"You are a debate evaluator. Evaluate the following argument on the topic '{topic}' on a scale of 0 to 10, considering strength, relevance, and persuasiveness. Provide a detailed explanation for your score."
    prompt = f"Argument: '{argument}'\n\nScore (0-10) and explanation:"
    return generate_llama(prompt, system)

def clean_topics(topics: List[str]) -> List[str]:
    cleaned_topics = []
    for topic in topics:
        topic = re.sub(r'^\d+\.\s*', '', topic.strip())
        topic = topic.replace('*', '').replace('"', '')
        topic = re.sub(r'^(Here are |Certainly! Here are |Sure, here are |The following are |These are )', '', topic, flags=re.IGNORECASE)
        topic = re.sub(r'^(five|[0-9]+) trending debate topics (that are |)suitable for a 10-minute debate session:?', '', topic, flags=re.IGNORECASE)
        topic = re.sub(r'^(five|[0-9]+) (trending |)debate topics.*:', '', topic, flags=re.IGNORECASE)
        topic = re.sub(r'^\d+\.\s*', '', topic)
        if topic and not topic.lower().startswith(("here are", "certainly", "sure,", "the following", "these are")):
            cleaned_topics.append(topic)
    return cleaned_topics

def generate_llm_suggestions(topic: str, position: str) -> List[str]:
    system = f"You are an AI debate assistant. Generate 3 strong argument suggestions for the {position} position on the topic: '{topic}'."
    prompt = "Generate 3 debate argument suggestions."
    response = generate_llama(prompt, system)
    return [suggestion.strip() for suggestion in response.split('\n') if suggestion.strip()]