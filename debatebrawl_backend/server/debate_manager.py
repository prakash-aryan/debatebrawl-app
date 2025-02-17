import random
import re
from typing import List, Dict, Tuple
from llm.llama_interface import generate_debate_topics, generate_argument, evaluate_argument, generate_llm_suggestions
from llm.gemma_interface import get_ai_opponent_response
from llm.phi_interface import get_debate_assistant_response
from genetic_algorithm.debate_strategy_evolver import DebateStrategyEvolver
from adversarial_search.debate_move_predictor import DebateMovePredictor

class DebateManager:

    def __init__(self):
        self.strategy_evolver = DebateStrategyEvolver(population_size=50, gene_length=3)
        self.move_predictor = DebateMovePredictor("Opening Statement")
        self.debates = {}

    def get_topics(self) -> List[str]:
        raw_topics = generate_debate_topics()
        cleaned_topics = []
        
        for topic in raw_topics:
            # Remove any leading numbers, dots, and whitespace
            topic = re.sub(r'^\d+\.?\s*', '', topic.strip())
            
            # Remove asterisks and quotation marks
            topic = topic.replace('*', '').replace('"', '')
            
            # Remove any explanatory text after the main topic
            topic = re.split(r'[.:]', topic)[0].strip()
            
            if topic and not topic.lower().startswith(("here are", "certainly", "sure,", "the following", "these are")):
                cleaned_topics.append(topic)
        
        # Remove any empty strings and get unique topics
        cleaned_topics = list(dict.fromkeys(filter(None, cleaned_topics)))
        
        # Ensure we have exactly 5 topics
        cleaned_topics = cleaned_topics[:5]
        while len(cleaned_topics) < 5:
            cleaned_topics.append(f"Placeholder Topic {len(cleaned_topics) + 1}")
        
        return cleaned_topics

    def start_debate(self, topic: str, user_position: str) -> str:
        debate_id = str(random.randint(1000, 9999))
        self.debates[debate_id] = {
            "topic": topic,
            "current_round": 1,
            "max_rounds": 5,
            "fitness_scores": [0.5 for _ in range(50)],
            "user_position": user_position,
            "ai_position": "against" if user_position == "for" else "for",
            "scores": {"user": 0, "ai": 0},
            "arguments": {"user": "", "ai": ""}
        }
        return debate_id

    def process_argument(self, debate_id: str, argument: str) -> Dict:
        debate = self.debates.get(debate_id)
        if not debate:
            raise ValueError("Debate not found")

        if debate["current_round"] > debate["max_rounds"]:
            raise ValueError("Debate has ended")

        user_score, user_feedback = self._parse_evaluation(evaluate_argument(argument, debate["topic"]))
        
        best_strategy = self.strategy_evolver.get_best_strategy(debate["fitness_scores"])
        ga_feedback = f"Focus on {best_strategy.genes[0]:.2f}% ethos, {best_strategy.genes[1]:.2f}% pathos, and {best_strategy.genes[2]:.2f}% logos."
        
        ai_argument = get_ai_opponent_response(debate["topic"], argument, debate["ai_position"])

        ai_score, ai_feedback = self._parse_evaluation(evaluate_argument(ai_argument, debate["topic"]))

        as_prediction = self.move_predictor.predict_best_move("Current State", 2)

        debate["fitness_scores"][debate["current_round"] - 1] = user_score
        self.strategy_evolver.evolve(debate["fitness_scores"])

        as_prediction_str = f"Predicted move: {as_prediction.move}, Confidence: {as_prediction.score:.2f}"

        llm_suggestions = self.generate_llm_suggestions(debate_id)

        debate_assistant_feedback = get_debate_assistant_response(debate["topic"], argument, debate["user_position"])

        # Update scores and arguments
        debate["scores"]["user"] += user_score
        debate["scores"]["ai"] += ai_score
        debate["arguments"]["user"] = argument
        debate["arguments"]["ai"] = ai_argument

        # Only increment the round here, not in the routes
        debate["current_round"] += 1

        return {
            "score": user_score,  # Return individual round score, not cumulative
            "ai_score": ai_score,  # Return individual round score, not cumulative
            "ai_argument": ai_argument,
            "ga_feedback": ga_feedback,
            "as_prediction": as_prediction_str,
            "llm_suggestions": llm_suggestions,
            "current_round": debate["current_round"],
            "max_rounds": debate["max_rounds"],
            "evaluation_feedback": user_feedback,
            "debate_assistant_feedback": debate_assistant_feedback,
            "arguments": debate["arguments"]
        }

    def generate_llm_suggestions(self, debate_id: str) -> List[str]:
        debate = self.debates.get(debate_id)
        if not debate:
            raise ValueError("Debate not found")
        return generate_llm_suggestions(debate["topic"], debate["user_position"])

    def _parse_evaluation(self, evaluation: str) -> Tuple[float, str]:
        try:
            # Find the score in the text
            score_match = re.search(r'(\d+(\.\d+)?)\s*(/|\s*out of\s*)\s*10', evaluation)
            if score_match:
                score = float(score_match.group(1))
            else:
                raise ValueError("Score not found in evaluation")

            # Normalize score to 0-1 range
            normalized_score = score / 10

            # Extract feedback (everything after the score)
            feedback = re.sub(r'^.*?(/|\s*out of\s*)\s*10', '', evaluation, flags=re.DOTALL).strip()

            return normalized_score, feedback
        except Exception as e:
            print(f"Error parsing evaluation: {e}")
            print(f"Raw evaluation: {evaluation}")
            return 0.5, "Error in evaluation"

    def get_debate_state(self, debate_id: str) -> Dict:
        debate = self.debates.get(debate_id)
        if not debate:
            raise ValueError("Debate not found")

        return {
            "topic": debate["topic"],
            "current_round": debate["current_round"],
            "max_rounds": debate["max_rounds"],
            "scores": debate["scores"],
            "arguments": debate["arguments"],
            "user_position": debate["user_position"],
            "ai_position": debate["ai_position"],
            "current_turn": "user" if debate["current_round"] % 2 != 0 else "ai",
            "status": "in_progress" if debate["current_round"] <= debate["max_rounds"] else "completed",
            "llm_suggestions": self.generate_llm_suggestions(debate_id),
            "ga_strategy": f"Focus on {debate['fitness_scores'][-1]:.2f}% ethos, {1-debate['fitness_scores'][-1]:.2f}% pathos/logos",
            "as_prediction": self.move_predictor.predict_best_move("Current State", 2).move,
        }