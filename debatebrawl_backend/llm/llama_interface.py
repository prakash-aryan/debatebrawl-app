import requests
import json
import re
from typing import List, Dict, Any, Optional

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_API_URL = "http://localhost:11434/api/chat"

def generate_llama(prompt: str, system: str = "", temperature: float = 0.7) -> str:
    """Generate text using the Llama model via Ollama"""
    data = {
        "model": "llama3.2:3b-instruct-q8_0",
        "prompt": prompt,
        "system": system,
        "temperature": temperature,
        "stream": False
    }
    
    response = requests.post(OLLAMA_API_URL, json=data)
    if response.status_code == 200:
        return response.json()['response']
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

def generate_debate_topics() -> List[str]:
    """Generate debate topics using Llama"""
    system = """You are a debate topic generator. Generate 5 concise, trending, 
    and controversial debate topics suitable for a 10-minute debate session. 
    Provide only the topics, without explanations or numbering."""
    
    prompt = "Generate 5 diverse and engaging debate topics."
    response = generate_llama(prompt, system)
    
    # Split into lines and clean
    topics = []
    for line in response.split('\n'):
        if line.strip():
            # Remove numbering, quotes, etc.
            clean_topic = re.sub(r'^\d+\.?\s*["]*', '', line.strip())
            clean_topic = clean_topic.rstrip('."')
            
            # Split by first sentence
            clean_topic = re.split(r'[.:]', clean_topic)[0].strip()
            
            if clean_topic and len(clean_topic) > 10:
                topics.append(clean_topic)
    
    # Return up to 5 unique topics
    unique_topics = list(dict.fromkeys(topics))
    return unique_topics[:5]

def generate_argument(topic: str, stance: str, 
                     strategy: Optional[Dict[str, Any]] = None) -> str:
    """Generate a debate argument with GA strategy integration"""
    # Base system prompt
    system = f"You are an AI debate assistant. Generate a strong argument {stance} the motion: '{topic}'."
    
    # Improve with GA strategy if provided
    if strategy:
        # Extract rhetorical balance
        ethos = strategy.get('ethos', 0.33)
        pathos = strategy.get('pathos', 0.33)
        logos = strategy.get('logos', 0.33)
        
        # Create strategic guidance based on rhetorical balance
        strategy_guidance = ""
        if ethos > 0.4:
            strategy_guidance += "Emphasize credibility and expertise. "
        if pathos > 0.4:
            strategy_guidance += "Include emotional appeals and values. "
        if logos > 0.4:
            strategy_guidance += "Focus on logical reasoning and evidence. "
        
        # Include specific tactics if available
        tactics = strategy.get('tactics', [])
        if tactics:
            tactics_str = ", ".join(tactics[:3])
            strategy_guidance += f"Use these debate tactics: {tactics_str}."
        
        system += f" Strategy: {strategy_guidance}"
    
    # Generate the argument
    prompt = f"Generate a compelling argument {stance} the topic: '{topic}'"
    
    # Adjust temperature based on strategy
    temperature = 0.7  # default
    if strategy:
        logos_weight = strategy.get('logos', 0.33)
        temperature = max(0.5, min(0.9, 0.7 + (0.3 - logos_weight)))
    
    return generate_llama(prompt, system, temperature)

def evaluate_argument(argument: str, topic: str, 
                     as_prediction: Optional[Dict[str, Any]] = None) -> str:
    """
    Evaluate an argument with consideration of debate moves and AS predictions
    
    Args:
        argument: The argument to evaluate
        topic: The debate topic
        as_prediction: Optional AS prediction data to inform evaluation
    """
    # Base system prompt
    system = f"""You are a debate evaluator. Evaluate the following argument on '{topic}' 
    on a scale of 0 to 10, considering strength, relevance, and persuasiveness. 
    Provide a detailed explanation for your score."""
    
    # Improve with AS predictions if available
    if as_prediction:
        detected_move = as_prediction.get('detected_move', '')
        if detected_move:
            system += f" The argument appears to use a '{detected_move}' approach."
        
        predicted_effectiveness = as_prediction.get('predicted_effectiveness', 0)
        if predicted_effectiveness:
            # Give a hint about expected effectiveness
            if predicted_effectiveness > 0.7:
                system += " This approach is likely to be highly effective in this context."
            elif predicted_effectiveness < 0.4:
                system += " This approach may not be optimal for this context."
    
    prompt = f"Argument: '{argument}'\n\nScore (0-10) and explanation:"
    return generate_llama(prompt, system)

def generate_llm_suggestions(topic: str, position: str, 
                           strategy: Optional[Dict[str, Any]] = None) -> List[str]:
    """Generate three distinct argument suggestions using structured JSON output"""
    
    # Define the system message with clear instructions for exactly three arguments
    system = f"""You are an AI debate assistant. Generate EXACTLY 3 distinct, strong argument suggestions for the {position} position on '{topic}'.
    Each suggestion must be a separate, complete paragraph of 3-5 sentences explaining a distinct argument approach.
    These must be three completely different argument approaches.
    Do not use placeholder text. Provide specific, concrete, substantive arguments.
    Do not include numbering, titles, or labels - just provide 3 complete paragraphs.
    """
    
    # Add strategy guidance if available
    if strategy:
        tactics = strategy.get('tactics', [])
        rhetorical_emphasis = []
        
        # Determine rhetorical emphasis
        if strategy.get('ethos', 0) > 0.4:
            rhetorical_emphasis.append("credibility-based")
        if strategy.get('pathos', 0) > 0.4:
            rhetorical_emphasis.append("emotional")
        if strategy.get('logos', 0) > 0.4:
            rhetorical_emphasis.append("logical")
        
        if rhetorical_emphasis:
            emphasis_str = " and ".join(rhetorical_emphasis)
            system += f" Emphasize {emphasis_str} arguments."
        
        if tactics:
            tactics_str = ", ".join(tactics[:3])
            system += f" Use these tactics: {tactics_str}."
    
    # Define a strict JSON schema to ensure exactly 3 separate suggestions
    output_schema = {
        "type": "object",
        "properties": {
            "suggestions": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "A complete paragraph argument suggestion"
                },
                "minItems": 3,
                "maxItems": 3
            }
        },
        "required": ["suggestions"]
    }
    
    # First try structured output for more reliable formatting
    try:
        data = {
            "model": "llama3.2:3b-instruct-q8_0",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Generate exactly 3 complete, substantive argument paragraphs for the {position} position on '{topic}'. Return as JSON."}
            ],
            "format": output_schema,
            "temperature": 0.6,  # Lower temperature for more consistent outputs
            "stream": False
        }
        
        response = requests.post(OLLAMA_CHAT_API_URL, json=data)
        if response.status_code == 200:
            content = response.json().get('message', {}).get('content', '{}')
            try:
                suggestions_data = json.loads(content)
                suggestions = suggestions_data.get('suggestions', [])
                
                # Verify we got exactly 3 substantial suggestions
                if len(suggestions) == 3 and all(len(s.strip()) > 20 for s in suggestions):
                    return suggestions
            except json.JSONDecodeError:
                pass  # Fall through to backup method
    except Exception as e:
        print(f"Structured output failed: {str(e)}. Using fallback method.")
    
    # Fallback: Use regular generation with specific format instructions
    fallback_system = f"""You are an AI debate assistant. Generate EXACTLY 3 distinct argument suggestions for the {position} position on '{topic}'.
    Format requirements:
    1. Each argument must be a separate paragraph.
    2. Paragraphs must be separated by TWO blank lines.
    3. Do NOT include any numbering, bullet points, or argument labels.
    4. Each paragraph should be 3-5 sentences.
    5. Each argument should take a different approach to the topic.
    
    Example format:
    First complete argument paragraph here. This is a complete thought with several sentences.
    
    
    Second complete argument paragraph here. This takes a different approach from the first.
    
    
    Third complete argument paragraph here. This takes yet another different approach.
    """
    
    prompt = f"Generate 3 distinct, persuasive argument paragraphs for the {position} position on '{topic}':"
    response = generate_llama(prompt, fallback_system, temperature=0.6)
    
    # Parse using the double newline separator we specified
    paragraphs = [p.strip() for p in response.split('\n\n\n') if p.strip()]
    
    # Alternative parsing if we don't get enough using the primary method
    if len(paragraphs) < 3:
        # Try other common separators
        paragraphs = [p.strip() for p in re.split(r'\n\n+|\n(?=\d+[\.\)])', response) if p.strip()]
        
        # Clean up any remaining formatting
        cleaned_paragraphs = []
        for paragraph in paragraphs:
            # Remove numbering, labels, etc.
            cleaned = re.sub(r'^\d+[\.\)]\s*', '', paragraph)
            cleaned = re.sub(r'\*\*?(.*?)\*\*?', r'\1', cleaned)
            cleaned = re.sub(r'^(Argument|Suggestion)\s*\d*\s*:?\s*', '', cleaned)
            cleaned = re.sub(r'^-\s*', '', cleaned)
            cleaned = cleaned.strip()
            
            if cleaned and len(cleaned) > 20:  # Only include substantial paragraphs
                cleaned_paragraphs.append(cleaned)
        
        paragraphs = cleaned_paragraphs
    
    # Ensure we have exactly 3 suggestions
    if len(paragraphs) < 3:
        # If we still don't have enough, generate individual arguments to fill the gap
        while len(paragraphs) < 3:
            single_prompt = f"Generate a single paragraph argument for the {position} position on '{topic}'. This should be different from previous arguments."
            single_response = generate_llama(single_prompt, "", temperature=0.7)
            cleaned = re.sub(r'^\d+[\.\)]\s*', '', single_response.strip())
            cleaned = re.sub(r'\*\*?(.*?)\*\*?', r'\1', cleaned)
            
            if cleaned and len(cleaned) > 20 and not any(cleaned in p for p in paragraphs):
                paragraphs.append(cleaned)
    
    return paragraphs[:3]  # Return only the first 3 to ensure consistent output