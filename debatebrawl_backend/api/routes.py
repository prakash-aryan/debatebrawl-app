from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
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

@router.post("/get_topics", response_model=TopicResponse)
async def api_get_topics(request: TopicRequest):
    topics = debate_manager.get_topics()
    return {"topics": topics}

@router.post("/start_debate")
async def start_debate(request: StartDebateRequest):
    db = firestore.client()
    user_ref = db.collection('users').document(request.user_id)
    user_data = user_ref.get().to_dict()

    if user_data['remainingFreeDebates'] <= 0:
        raise HTTPException(status_code=403, detail="No free debates remaining")

    debate_id = debate_manager.start_debate(request.topic, request.position)
    user_name = user_data.get('name', 'User')

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

    return {"debate_id": debate_id, "status": "started"}

@router.post("/submit_argument", response_model=ArgumentResponse)
async def submit_argument(request: ArgumentRequest):
    result = debate_manager.process_argument(request.debate_id, request.argument)

    db = firestore.client()
    debate_ref = db.collection('debates').document(request.debate_id)
    debate_data = debate_ref.get().to_dict()

    current_round = result['current_round'] - 1  # Use the round before increment

    debate_ref.update({
        f'arguments.round_{current_round}.user': request.argument,
        f'arguments.round_{current_round}.ai': result['ai_argument'],
        'current_round': result['current_round'],  # Use the new round number
        'current_turn': 'user',
        'scores.user': firestore.Increment(result['score']),
        'scores.ai': firestore.Increment(result['ai_score']),
        'llm_suggestions': result['llm_suggestions'],
        'ga_strategy': result['ga_feedback'],
        'as_prediction': result['as_prediction'],
        f'evaluation_feedback.round_{current_round}': result['evaluation_feedback']
    })

    # Fetch updated debate data
    updated_debate_data = debate_ref.get().to_dict()

    if current_round >= 5:  # Assuming 5 rounds per debate
        await finalize_debate(request.debate_id, request.user_id)

    return ArgumentResponse(
        score=result['score'],
        ai_score=result['ai_score'],
        ai_argument=result['ai_argument'],
        ga_feedback=result['ga_feedback'],
        as_prediction=result['as_prediction'],
        llm_suggestions=result['llm_suggestions'],
        evaluation_feedback=result['evaluation_feedback'],
        debate_assistant_feedback=result['debate_assistant_feedback'],
        current_round=updated_debate_data['current_round'],
        max_rounds=5,
        arguments=updated_debate_data.get('arguments', {})
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
        llm_suggestions = debate_manager.generate_llm_suggestions(debate_id)
        debate_ref.update({'llm_suggestions': llm_suggestions})
    else:
        llm_suggestions = debate_data['llm_suggestions']

    # Get the latest evaluation feedback
    current_round = debate_data.get('current_round', 1)
    latest_evaluation = debate_data.get('evaluation_feedback', {}).get(f'round_{current_round - 1}', '')

    return DebateStateResponse(
        status=debate_data.get('status', 'unknown'),
        current_round=current_round,
        current_turn=debate_data.get('current_turn', 'user'),
        scores=debate_data.get('scores', {}),
        arguments=debate_data.get('arguments', {}),
        topic=debate_data.get('topic', ''),
        llm_suggestions=llm_suggestions,
        ga_strategy=debate_data.get('ga_strategy'),
        as_prediction=debate_data.get('as_prediction'),
        user_position=debate_data.get('user_position', ''),
        ai_position=debate_data.get('ai_position', ''),
        evaluation_feedback=latest_evaluation
    )

@router.get("/user_stats/{user_id}")
async def get_user_stats(user_id: str):
    db = firestore.client()
    user_ref = db.collection('users').document(user_id)
    user_data = user_ref.get().to_dict()

    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "totalDebates": user_data.get('totalDebates', 0),
        "remainingFreeDebates": user_data.get('remainingFreeDebates', 5),
        "wins": user_data.get('wins', 0),
        "losses": user_data.get('losses', 0),
        "draws": user_data.get('draws', 0)
    }

@router.get("/user_debate_history/{user_id}")
async def get_user_debate_history(user_id: str):
    db = firestore.client()
    history_ref = db.collection('users').document(user_id).collection('debateHistory')
    history = history_ref.order_by('date', direction=firestore.Query.DESCENDING).limit(10).stream()

    return [doc.to_dict() for doc in history]

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

    # Add debate to user's debate history
    user_ref.collection('debateHistory').add({
        'debateId': debate_id,
        'topic': debate_data['topic'],
        'date': firestore.SERVER_TIMESTAMP,
        'result': result,
        'userScore': user_score,
        'aiScore': ai_score
    })