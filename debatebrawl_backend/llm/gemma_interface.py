import requests
import json
import re
from typing import List, Dict, Any, Optional

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_API_URL = "http://localhost:11434/api/chat"

def generate_gemma(prompt: str, system: str = "", temperature: float = 0.7) -> str:
    """Generate text using the Gemma model via Ollama"""
    data = {
        "model": "gemma2:2b-instruct-q8_0",
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

def get_ai_opponent_response(topic: str, user_argument: str, ai_position: str, 
                            strategy: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate an AI opponent response using the GA strategy and AS predictions
    
    Args:
        topic: The debate topic
        user_argument: The user's argument text
        ai_position: The AI's position (for/against)
        strategy: Dict with GA strategy and AS predictions
    """
    # Base system prompt
    system = f"""You are an AI debater taking the {ai_position} position on '{topic}'. 
    Provide a clear, coherent counter-argument to the user's points without formatting 
    markers like asterisks or bullet points. Be persuasive and respond directly to 
    the user's argument."""
    
    # Improve with GA and AS strategies if provided
    if strategy:
        # Apply rhetorical balance from GA
        ethos_weight = strategy.get('ethos', 0.33)
        pathos_weight = strategy.get('pathos', 0.33)
        logos_weight = strategy.get('logos', 0.33)
        
        # Determine strategic emphasis
        rhetorical_guidance = ""
        if max(ethos_weight, pathos_weight, logos_weight) > 0.4:
            if ethos_weight > pathos_weight and ethos_weight > logos_weight:
                rhetorical_guidance = "Emphasize your credibility and expertise. Reference your knowledge and experience."
            elif pathos_weight > ethos_weight and pathos_weight > logos_weight:
                rhetorical_guidance = "Appeal to emotions and values. Connect with the audience's feelings and concerns."
            else:
                rhetorical_guidance = "Focus on logical reasoning and evidence. Present clear arguments supported by facts."
        
        # Use tactics from GA
        tactics_guidance = ""
        tactics = strategy.get('tactics', [])
        if tactics:
            # Map tactics to specific instructions
            tactics_map = {
                "present_evidence": "cite specific evidence, studies, or statistics",
                "appeal_to_emotion": "appeal to emotions like hope, fear, or compassion",
                "logical_reasoning": "use clear logical structure with premises and conclusions",
                "address_counterarguments": "preemptively address potential counterarguments",
                "provide_examples": "include concrete, relatable examples",
                "use_analogy": "use an analogy or metaphor to illustrate your point",
                "cite_expert_opinion": "reference expert opinions or authorities",
                "highlight_consequences": "emphasize likely consequences or outcomes",
                "reframe_issue": "reframe the issue from a different perspective",
                "ask_rhetorical_questions": "include thought-provoking rhetorical questions",
                "tell_story": "include a brief illustrative story or scenario",
                "establish_credibility": "establish your credibility on this topic",
                "concede_points": "acknowledge valid points before countering",
                "build_common_ground": "identify areas of common ground before presenting differences"
            }
            
            # Select tactics to include in prompt
            selected_tactics = [tactics_map.get(tactic, tactic) for tactic in tactics[:3] 
                              if tactic in tactics_map]
            
            if selected_tactics:
                tactics_guidance = f"In your response, {', '.join(selected_tactics)}. "
        
        # Incorporate AS prediction and counter strategy
        as_guidance = ""
        if 'predicted_user_move' in strategy:
            predicted_move = strategy['predicted_user_move']
            as_guidance = f"The user is likely using a '{predicted_move}' approach. "
            
            # Add counter strategy if available
            if 'recommended_counter' in strategy:
                counter = strategy['recommended_counter']
                as_guidance += f"To effectively counter this, use a '{counter}' approach. "
        
        # Combine all guidance
        strategy_guidance = f"{rhetorical_guidance} {tactics_guidance}{as_guidance}"
        system += f" {strategy_guidance}Keep your response focused and persuasive. Do not use formatting marks like asterisks or bullet points."
    
    prompt = f"""User's argument on topic '{topic}': {user_argument}

Your response as the opponent arguing {ai_position} the topic (provide a clear, persuasive argument without any formatting or bullet points):"""
    
    # Adjust temperature based on strategy emphasis
    temperature = 0.7  # default
    if strategy:
        # More logos = lower temperature (more precise)
        # More pathos = higher temperature (more creative)
        logos_influence = strategy.get('logos', 0.33)
        pathos_influence = strategy.get('pathos', 0.33)
        temperature = 0.5 + pathos_influence - (logos_influence * 0.3)
    
    response = generate_gemma(prompt, system, temperature)
    
    # Clean the response
    cleaned_response = re.sub(r'\*\*?(.*?)\*\*?', r'\1', response)  # Remove bold/italic
    cleaned_response = re.sub(r'^\s*-\s+', '', cleaned_response, flags=re.MULTILINE)  # Remove bullet points
    cleaned_response = re.sub(r'^\s*\d+[\.\)]\s+', '', cleaned_response, flags=re.MULTILINE)  # Remove numbering
    
    return cleaned_response