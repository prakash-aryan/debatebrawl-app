import requests
import json
import re
from typing import Dict, Any, Optional, List

OLLAMA_API_URL = "http://localhost:11434/api/generate"

def generate_phi(prompt: str, system: str = "", temperature: float = 0.7) -> str:
    """Generate text using the Phi model via Ollama"""
    data = {
        "model": "phi3.5:3.8b-mini-instruct-q5_K_M",
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

def get_debate_assistant_response(topic: str, user_argument: str, user_position: str,
                                 ga_strategy: Optional[Dict[str, Any]] = None,
                                 as_prediction: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate debate assistant feedback integrating insights from GA and AS
    
    Args:
        topic: The debate topic
        user_argument: The user's argument
        user_position: User's stance (for/against)
        ga_strategy: Optional GA-evolved strategy data
        as_prediction: Optional AS prediction data
    """
    # Base system prompt
    system = f"""You are an AI debate assistant helping a user argue for the {user_position} 
    position on: '{topic}'. Provide constructive feedback and suggestions to improve 
    their argument. Focus on educational value and skill development.
    Do not use formatting marks like asterisks or bullet points."""
    
    # Improve with GA and AS insights
    feedback_guidance = ""
    
    # Integrate GA strategy insights if available
    if ga_strategy:
        # Extract rhetorical recommendations from GA
        ethos = ga_strategy.get('ethos', 0.33)
        pathos = ga_strategy.get('pathos', 0.33)
        logos = ga_strategy.get('logos', 0.33)
        
        # Identify areas for improvement based on GA strategy
        improvements = []
        if ethos > 0.4 and 'ethos' not in (ga_strategy.get('strengths') or []):
            improvements.append("establishing more credibility")
        if pathos > 0.4 and 'pathos' not in (ga_strategy.get('strengths') or []):
            improvements.append("incorporating more emotional appeal")
        if logos > 0.4 and 'logos' not in (ga_strategy.get('strengths') or []):
            improvements.append("strengthening logical reasoning")
        
        if improvements:
            feedback_guidance += f"Focus on {', '.join(improvements)}. "
        
        # Include recommended tactics from GA
        tactics = ga_strategy.get('tactics', [])
        if tactics:
            feedback_guidance += f"Consider using these tactics: {', '.join(tactics[:3])}. "
    
    # Integrate AS predictions if available
    if as_prediction:
        detected_move = as_prediction.get('detected_move', '')
        if detected_move:
            feedback_guidance += f"You seem to be using a '{detected_move}' approach. "
        
        # Include AS counter prediction
        predicted_counter = as_prediction.get('predicted_counter', '')
        if predicted_counter:
            feedback_guidance += f"Be prepared for your opponent to use a '{predicted_counter}' counter. "
        
        # Include effectiveness prediction
        effectiveness = as_prediction.get('predicted_effectiveness', 0)
        if effectiveness > 0.7:
            feedback_guidance += "Your current approach is likely to be effective. "
        elif effectiveness < 0.4:
            feedback_guidance += "Consider altering your approach for more effectiveness. "
    
    # Add feedback guidance to system prompt
    if feedback_guidance:
        system += f" {feedback_guidance}Provide specific, actionable advice that focuses on educational value and debate skill improvement."
    
    prompt = f"""
Topic: {topic}
User's position: {user_position}
User's argument: {user_argument}

Please provide constructive feedback that:
1. Identifies strengths in the argument
2. Suggests specific improvements
3. Offers educational guidance on debate techniques
4. Helps prepare for potential counter-arguments

Your feedback (without bullet points or formatting):
"""
    
    # Set temperature based on needed specificity
    temperature = 0.7
    if ga_strategy and len(ga_strategy.get('tactics', [])) > 0:
        # More specific tactics require lower temperature
        temperature = 0.6
    
    response = generate_phi(prompt, system, temperature)
    
    # Clean the response
    cleaned_response = re.sub(r'\*\*?(.*?)\*\*?', r'\1', response)  # Remove bold/italic
    cleaned_response = re.sub(r'^\s*-\s+', '', cleaned_response, flags=re.MULTILINE)  # Remove bullet points
    cleaned_response = re.sub(r'^\s*\d+[\.\)]\s+', '', cleaned_response, flags=re.MULTILINE)  # Remove numbering
    
    return cleaned_response

def suggest_improvements(topic: str, argument: str, position: str,
                       ga_strategy: Optional[Dict[str, Any]] = None,
                       as_prediction: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Suggest targeted improvements based on GA strategy and AS predictions
    
    Args:
        topic: The debate topic
        argument: The user's argument
        position: User's stance (for/against)
        ga_strategy: Optional GA strategy data
        as_prediction: Optional AS prediction data
    
    Returns:
        List of improvement suggestions
    """
    # Combine GA and AS insights for better suggestions
    strategy_context = ""
    
    if ga_strategy:
        # Extract key tactics from GA
        tactics = ga_strategy.get('tactics', [])
        if tactics:
            strategy_context += f"Consider these tactics: {', '.join(tactics[:3])}. "
    
    if as_prediction:
        # Include opponent prediction from AS
        predicted_move = as_prediction.get('predicted_user_move', '')
        if predicted_move:
            strategy_context += f"Your opponent may use a '{predicted_move}' approach. "
        
        # Include recommended counter
        counter = as_prediction.get('recommended_counter', '')
        if counter:
            strategy_context += f"Consider countering with '{counter}'. "
    
    system = f"""You are an AI debate coach and educator. The topic is '{topic}' and the position is '{position}'. 
    Your role is to provide educational guidance and feedback to improve debate skills.
    Suggest 3 specific improvements to strengthen this argument. {strategy_context}
    Provide each suggestion as a clear, specific, educational paragraph. Focus on teaching debate principles and techniques.
    Do not use bullet points, numbering, or formatting markers."""
    
    prompt = f"""
Debate Topic: {topic}
Position: {position}
Argument: {argument}

Please provide exactly 3 educational improvement suggestions that:
1. Are specific and actionable
2. Teach debate techniques
3. Address different aspects of effective argumentation
4. Help prepare for the opponent's potential responses

Provide your 3 suggestions as separate paragraphs:
"""
    
    response = generate_phi(prompt, system)
    
    # Clean and split the response into paragraphs
    cleaned_text = re.sub(r'\*\*?(.*?)\*\*?', r'\1', response)
    cleaned_text = re.sub(r'^\s*-\s+', '', cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'^\s*\d+[\.\)]\s+', '', cleaned_text, flags=re.MULTILINE)
    
    # Split by double newlines or paragraph indicators
    paragraphs = re.split(r'\n\n+|(?=\n\s*\d+[\.\)])|(?=\n\s*Suggestion\s*\d*\s*:)', cleaned_text)
    
    suggestions = []
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
            
        # Clean up any formatting or indicators
        cleaned = re.sub(r'^\s*\d+[\.\)]\s*', '', paragraph.strip())
        cleaned = re.sub(r'^(Suggestion|Improvement)\s*\d*\s*:?\s*', '', cleaned)
        cleaned = cleaned.strip()
        
        if cleaned and len(cleaned) > 20:  # Only include substantial suggestions
            suggestions.append(cleaned)
    
    # If we still don't have 3 suggestions, generate additional ones
    while len(suggestions) < 3:
        additional_prompt = f"Provide a single educational suggestion to improve a debate argument on '{topic}' from the {position} position."
        additional_response = generate_phi(additional_prompt, "You are a debate coach. Be concise and educational.")
        cleaned = re.sub(r'^\d+[\.\)]\s*', '', additional_response.strip())
        cleaned = re.sub(r'\*\*?(.*?)\*\*?', r'\1', cleaned)
        
        if cleaned and len(cleaned) > 20 and not any(cleaned in s for s in suggestions):
            suggestions.append(cleaned)
    
    # Ensure we have exactly 3 suggestions
    return suggestions[:3]