const API_BASE_URL = 'http://localhost:8000/api';

export async function getTopics(userId: string): Promise<string[]> {
  const response = await fetch(`${API_BASE_URL}/get_topics`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ user_id: userId }),
  });
  if (!response.ok) {
    throw new Error('Failed to fetch topics');
  }
  const data = await response.json();
  return data.topics;
}

export async function startDebate(userId: string, topic: string, position: string): Promise<{ debate_id: string; status: string }> {
  const response = await fetch(`${API_BASE_URL}/start_debate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ user_id: userId, topic, position }),
  });
  if (!response.ok) {
    throw new Error('Failed to start debate');
  }
  return response.json();
}

export async function getDebateState(debateId: string): Promise<{
  status: string;
  current_round: number;
  current_turn: string;
  scores: { user: number; ai: number };
  arguments: { user: string; ai: string };
  topic: string;
  llm_suggestions: string[];
  ga_strategy: string | null;
  as_prediction: string | null;
  time_left: number;
  start_time: number;
  user_position: string;
  ai_position: string;
  evaluation_feedback: string;
}> {
  const response = await fetch(`${API_BASE_URL}/debate_state/${debateId}`);
  if (!response.ok) {
    throw new Error('Failed to fetch debate state');
  }
  return response.json();
}

export async function submitArgument(debateId: string, userId: string, argument: string): Promise<{
  score: number;
  ai_score: number;
  ai_argument: string;
  ga_feedback: string;
  as_prediction: string;
  llm_suggestions: string[];
  evaluation_feedback: string;
  current_round: number;
  max_rounds: number;
  arguments: { user: string; ai: string };
}> {
  const response = await fetch(`${API_BASE_URL}/submit_argument`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ debate_id: debateId, user_id: userId, argument }),
  });
  if (!response.ok) {
    throw new Error('Failed to submit argument');
  }
  return response.json();
}

export async function getUserStats(userId: string): Promise<{
  totalDebates: number;
  remainingFreeDebates: number;
  wins: number;
  losses: number;
  draws: number;
}> {
  const response = await fetch(`${API_BASE_URL}/user_stats/${userId}`);
  if (!response.ok) {
    throw new Error('Failed to fetch user stats');
  }
  return response.json();
}

export async function getUserDebateHistory(userId: string): Promise<Array<{
  debateId: string;
  topic: string;
  date: string;
  result: 'win' | 'loss' | 'draw';
  userScore: number;
  aiScore: number;
}>> {
  const response = await fetch(`${API_BASE_URL}/user_debate_history/${userId}`);
  if (!response.ok) {
    throw new Error('Failed to fetch user debate history');
  }
  return response.json();
}

export default {
  getTopics,
  startDebate,
  getDebateState,
  submitArgument,
  getUserStats,
  getUserDebateHistory,
};