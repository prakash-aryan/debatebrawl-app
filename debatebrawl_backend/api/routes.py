from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import re
from firebase_admin import firestore
from server.debate_manager import DebateManager

router = APIRouter()
debate_manager = DebateManager()


class TopicRequest(BaseModel):
    user_id: str


class TopicResponse(BaseModel):
    topics: List[str]


class StartDebateRequest(BaseModel):
    user_id: str
    topic: str
    position: str


class ArgumentRequest(BaseModel):
    user_id: str
    debate_id: str
    argument: str


class ArgumentResponse(BaseModel):
    score: float
    ai_score: float
    ai_argument: str
    ga_feedback: str
    as_prediction: str
    llm_suggestions: List[str]
    evaluation_feedback: str
    debate_assistant_feedback: str
    current_round: int
    max_rounds: int
    arguments: Dict[str, Dict[str, str]]


class DebateStateResponse(BaseModel):
    status: str
    current_round: int
    current_turn: str
    scores: Dict[str, float]
    arguments: Dict[str, Dict[str, str]]
    topic: str
    llm_suggestions: List[str]
    ga_strategy: Optional[str]
    as_prediction: Optional[str]
    user_position: str
    ai_position: str
    evaluation_feedback: Optional[str]


def clean_markdown(text: str) -> str:
    """
    Clean markdown formatting from text while enhancing readability with proper line breaks
    """
    if not text:
        return ""
        
    # Replace markdown bold/italic
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    
    # Clean bullet points
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    
    # Clean numbered lists
    text = re.sub(r'^\s*\d+[\.\)]\s+', '', text, flags=re.MULTILINE)
    
    # Add line breaks before section headers
    # Match section headers like "Strength (4/5):" or "Relevance:"
    text = re.sub(r'([.!?])\s+(Strength|Weakness|Relevance|Persuasiveness|Overall|Emotional appeal)', r'\1\n\n\2', text)
    
    # Add line breaks for rating sections
    text = re.sub(r'(\([0-9]+/[0-9]+\):)', r'\n\n\1', text)
    
    # Format sections within the AI response with paragraph breaks
    text = re.sub(r'([.!?])\s+(Consider|Furthermore|Instead|Additionally|Moreover|It\'s about)', r'\1\n\n\2', text)
    
    # Add paragraph breaks after sentences that end sections (improve readability in AI's response)
    text = re.sub(r'([.!?])\s+([A-Z][a-z]+\s+[a-z]+)', r'\1\n\n\2', text, flags=re.MULTILINE)
    
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Clean leading/trailing whitespace
    text = text.strip()
    
    return text

def clean_suggestions(suggestions: List[str]) -> List[str]:
    """Clean a list of suggestions"""
    cleaned = []
    for suggestion in suggestions:
        # Remove numbering and formatting symbols
        clean = re.sub(r'^\s*\d+[\.\)]\s*', '', suggestion)
        clean = clean_markdown(clean)
        
        # Ensure the first letter is capitalized
        if clean and len(clean) > 0:
            clean = clean[0].upper() + clean[1:] if len(clean) > 1 else clean[0].upper()
            cleaned.append(clean)
    
    return cleaned


@router.post("/get_topics", response_model=TopicResponse)
async def api_get_topics(request: TopicRequest):
    topics = debate_manager.get_topics()
    return {"topics": topics}


@router.post("/start_debate")
async def start_debate(request: StartDebateRequest):
    db = firestore.client()
    user_ref = db.collection('users').document(request.user_id)
    user_data = user_ref.get().to_dict()

    if not user_data:
        # Create user if not exists
        user_ref.set({
            'name': 'User',
            'remainingFreeDebates': 45,
            'totalDebates': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0
        })
    elif user_data.get('remainingFreeDebates', 0) <= 0:
        raise HTTPException(status_code=403, detail="No free debates remaining")

    debate_id = debate_manager.start_debate(request.topic, request.position)
    user_name = user_data.get('name', 'User') if user_data else 'User'

    debate_ref = db.collection('debates').document(debate_id)
    debate_ref.set({
        'topic': request.topic,
        'user_id': request.user_id,
        'user_name': user_name,
        'status': 'in_progress',
        'current_round': 1,
        'current_turn': 'user',
        'scores': {'user': 0, 'ai': 0},
        'arguments': {},
        'user_position': request.position,
        'ai_position': 'against' if request.position == 'for' else 'for',
        'evaluation_feedback': {}
    })

    user_ref.update({
        'remainingFreeDebates': firestore.Increment(-1),
        'totalDebates': firestore.Increment(1)
    })

    # Generate initial suggestions for the first round
    initial_suggestions = debate_manager.generate_llm_suggestions(debate_id)
    debate_ref.update({
        'llm_suggestions': clean_suggestions(initial_suggestions)
    })

    return {"debate_id": debate_id, "status": "started"}


@router.post("/submit_argument", response_model=ArgumentResponse)
async def submit_argument(request: ArgumentRequest):
    result = debate_manager.process_argument(request.debate_id, request.argument)

    # Clean LLM outputs
    ai_argument = clean_markdown(result['ai_argument'])
    ga_feedback = clean_markdown(result['ga_feedback'])
    evaluation_feedback = clean_markdown(result['evaluation_feedback'])
    debate_assistant_feedback = clean_markdown(result['debate_assistant_feedback'])
    llm_suggestions = clean_suggestions(result['llm_suggestions'])

    # Clean AS prediction
    as_prediction = result['as_prediction']
    if as_prediction:
        as_prediction = re.sub(r'Predicted move: ', '', as_prediction)
        as_prediction = re.sub(r', Confidence: \d+\.\d+', '', as_prediction)

    db = firestore.client()
    debate_ref = db.collection('debates').document(request.debate_id)
    debate_data = debate_ref.get().to_dict()

    current_round = result['current_round'] - 1  # Use the round before increment

    debate_ref.update({
        f'arguments.round_{current_round}.user': request.argument,
        f'arguments.round_{current_round}.ai': ai_argument,
        'current_round': result['current_round'],  # Use the new round number
        'current_turn': 'user',
        'scores.user': firestore.Increment(result['score']),
        'scores.ai': firestore.Increment(result['ai_score']),
        'llm_suggestions': llm_suggestions,
        'ga_strategy': ga_feedback,
        'as_prediction': as_prediction,
        f'evaluation_feedback.round_{current_round}': evaluation_feedback
    })

    # Fetch updated debate data
    updated_debate_data = debate_ref.get().to_dict()

    if current_round >= 5:  # Assuming 5 rounds per debate
        await finalize_debate(request.debate_id, request.user_id)

    # Format arguments for response - FIXED
    arguments = {
        f"round_{current_round}": {
            "user": request.argument,
            "ai": ai_argument
        }
    }

    return ArgumentResponse(
        score=result['score'],
        ai_score=result['ai_score'],
        ai_argument=ai_argument,
        ga_feedback=ga_feedback,
        as_prediction=as_prediction,
        llm_suggestions=llm_suggestions,
        evaluation_feedback=evaluation_feedback,
        debate_assistant_feedback=debate_assistant_feedback,
        current_round=updated_debate_data['current_round'],
        max_rounds=5,
        arguments=arguments
    )


@router.get("/debate_state/{debate_id}", response_model=DebateStateResponse)
async def get_debate_state(debate_id: str):
    db = firestore.client()
    debate_ref = db.collection('debates').document(debate_id)
    debate = debate_ref.get()

    if not debate.exists:
        raise HTTPException(status_code=404, detail="Debate not found")

    debate_data = debate.to_dict()

    # Generate new suggestions if they don't exist
    if not debate_data.get('llm_suggestions'):
        raw_suggestions = debate_manager.generate_llm_suggestions(debate_id)
        llm_suggestions = clean_suggestions(raw_suggestions)
        debate_ref.update({'llm_suggestions': llm_suggestions})
    else:
        llm_suggestions = debate_data['llm_suggestions']

    # Clean GA strategy
    ga_strategy = debate_data.get('ga_strategy')
    if ga_strategy:
        ga_strategy = clean_markdown(ga_strategy)

    # Clean AS prediction
    as_prediction = debate_data.get('as_prediction')
    if as_prediction:
        as_prediction = clean_markdown(as_prediction)

    # Get arguments with cleaned AI responses
    arguments = debate_data.get('arguments', {})
    cleaned_arguments = {}
    
    for round_key, round_args in arguments.items():
        cleaned_arguments[round_key] = {
            "user": round_args.get("user", ""),
            "ai": clean_markdown(round_args.get("ai", ""))
        }

    # Get the latest evaluation feedback
    current_round = debate_data.get('current_round', 1)
    evaluation_feedback = debate_data.get('evaluation_feedback', {}).get(f'round_{current_round - 1}', '')
    if evaluation_feedback:
        evaluation_feedback = clean_markdown(evaluation_feedback)

    return DebateStateResponse(
        status=debate_data.get('status', 'unknown'),
        current_round=current_round,
        current_turn=debate_data.get('current_turn', 'user'),
        scores=debate_data.get('scores', {}),
        arguments=cleaned_arguments,
        topic=debate_data.get('topic', ''),
        llm_suggestions=llm_suggestions,
        ga_strategy=ga_strategy,
        as_prediction=as_prediction,
        user_position=debate_data.get('user_position', ''),
        ai_position=debate_data.get('ai_position', ''),
        evaluation_feedback=evaluation_feedback
    )


@router.get("/user_stats/{user_id}")
async def get_user_stats(user_id: str):
    db = firestore.client()
    user_ref = db.collection('users').document(user_id)
    user_data = user_ref.get().to_dict()

    if not user_data:
        # Create default user if not exists
        default_user = {
            "totalDebates": 0,
            "remainingFreeDebates": 45,
            "wins": 0,
            "losses": 0,
            "draws": 0
        }
        user_ref.set(default_user)
        return default_user

    return {
        "totalDebates": user_data.get('totalDebates', 0),
        "remainingFreeDebates": user_data.get('remainingFreeDebates', 45),
        "wins": user_data.get('wins', 0),
        "losses": user_data.get('losses', 0),
        "draws": user_data.get('draws', 0)
    }


@router.get("/user_debate_history/{user_id}")
async def get_user_debate_history(user_id: str):
    db = firestore.client()
    history_ref = db.collection('users').document(user_id).collection('debateHistory')
    history = history_ref.order_by('date', direction=firestore.Query.DESCENDING).limit(10).stream()

    # Clean debate topics
    history_data = []
    for doc in history:
        data = doc.to_dict()
        if 'topic' in data:
            # Keep topic but ensure it's clean
            data['topic'] = clean_markdown(data['topic'])
        history_data.append(data)

    return history_data


@router.get("/trending_topics")
async def get_trending_topics():
    """Get a list of trending debate topics"""
    topics = debate_manager.get_topics()
    return {"topics": topics}


@router.get("/debate_analytics/{debate_id}")
async def get_debate_analytics(debate_id: str):
    """Get detailed analytics for a specific debate"""
    db = firestore.client()
    debate_ref = db.collection('debates').document(debate_id)
    debate = debate_ref.get()

    if not debate.exists:
        raise HTTPException(status_code=404, detail="Debate not found")

    debate_data = debate.to_dict()
    
    # Extract GA and AS data if available
    evolution_history = []
    as_predictions = []
    
    for round_num in range(1, debate_data.get('current_round', 1)):
        round_key = f'round_{round_num}'
        if round_key in debate_data.get('arguments', {}):
            evolution_history.append({
                "round": round_num,
                "strategy": clean_markdown(debate_data.get('ga_strategy', ''))
            })
            
    # Return clean analytics data
    return {
        "topic": debate_data.get('topic', ''),
        "total_rounds": debate_data.get('current_round', 1) - 1,
        "final_scores": debate_data.get('scores', {}),
        "evolution_history": evolution_history,
        "most_effective_tactics": ["logical_reasoning", "present_evidence", "address_counterarguments"]
    }


async def finalize_debate(debate_id: str, user_id: str):
    db = firestore.client()
    debate_ref = db.collection('debates').document(debate_id)
    debate_data = debate_ref.get().to_dict()

    user_ref = db.collection('users').document(user_id)
    
    user_score = debate_data['scores']['user']
    ai_score = debate_data['scores']['ai']
    
    result = 'draw'
    if user_score > ai_score:
        result = 'win'
    elif user_score < ai_score:
        result = 'loss'

    # Update debate document
    debate_ref.update({
        'status': 'completed',
        'result': result
    })

    # Update user document
    if result == 'win':
        user_ref.update({
            'wins': firestore.Increment(1)
        })
    elif result == 'loss':
        user_ref.update({
            'losses': firestore.Increment(1)
        })
    elif result == 'draw':
        user_ref.update({
            'draws': firestore.Increment(1)
        })

    # Add debate to user's debate history with cleaned topic
    topic = clean_markdown(debate_data['topic'])
    user_ref.collection('debateHistory').add({
        'debateId': debate_id,
        'topic': topic,
        'date': firestore.SERVER_TIMESTAMP,
        'result': result,
        'userScore': user_score,
        'aiScore': ai_score
    })