import random
import re
from typing import List, Dict, Tuple, Any, Optional
from llm.llama_interface import generate_debate_topics, generate_argument, evaluate_argument, generate_llm_suggestions
from llm.gemma_interface import get_ai_opponent_response
from llm.phi_interface import get_debate_assistant_response, suggest_improvements
from genetic_algorithm.debate_strategy_evolver import DebateStrategyEvolver, Strategy
from adversarial_search.debate_move_predictor import DebateMovePredictor, Move

class DebateManager:
    """
    Central manager integrating GA and AS components with LLMs for adaptive debate
    """
    def __init__(self):
        # Initialize GA strategy evolver with reasonable population size
        self.strategy_evolver = DebateStrategyEvolver(population_size=20)
        
        # Initialize AS move predictor
        self.move_predictor = DebateMovePredictor("Opening Statement")
        
        # Store active debates
        self.debates = {}
        
        # Generations per debate round 
        self.generations_per_round = 3
        
        # How strongly GA/AS influence the AI responses (0-1)
        self.ga_influence = 0.8
        self.as_influence = 0.7

    def get_topics(self) -> List[str]:
        """Generate and clean debate topics using LLama"""
        raw_topics = generate_debate_topics()
        cleaned_topics = []
        
        for topic in raw_topics:
            # Remove numbering and formatting
            topic = re.sub(r'^\d+\.?\s*', '', topic.strip())
            topic = topic.replace('*', '').replace('"', '')
            topic = re.split(r'[.:]', topic)[0].strip()
            
            if topic and not topic.lower().startswith(("here are", "certainly", "sure,", "the following", "these are")):
                cleaned_topics.append(topic)
        
        # Ensure we have exactly 5 topics
        cleaned_topics = list(dict.fromkeys(filter(None, cleaned_topics)))
        cleaned_topics = cleaned_topics[:5]
        while len(cleaned_topics) < 5:
            cleaned_topics.append(f"Topic {len(cleaned_topics) + 1}")
        
        return cleaned_topics

    def start_debate(self, topic: str, user_position: str) -> str:
        """Initialize a new debate with integrated GA and AS components"""
        debate_id = str(random.randint(1000, 9999))
        
        # Create diverse initial population for GA
        initial_fitness = [random.uniform(0.4, 0.6) for _ in range(self.strategy_evolver.population_size)]
        
        # Set up debate state
        self.debates[debate_id] = {
            "topic": topic,
            "current_round": 1,
            "max_rounds": 5,
            "user_position": user_position,
            "ai_position": "against" if user_position == "for" else "for",
            "scores": {"user": 0, "ai": 0},
            "arguments": {"user": {}, "ai": {}},
            "move_history": [],
            
            # GA state
            "ga_fitness": initial_fitness.copy(),
            "best_strategy": None,
            "evolution_history": [],
            
            # AS state
            "as_predictions": [],
            "detected_moves": [],
            "counter_moves": [],
            
            # Feedback and analysis
            "evaluations": {"user": {}, "ai": {}},
            "suggestions": []
        }
        
        return debate_id

    def process_argument(self, debate_id: str, argument: str) -> Dict:
        """
        Process a user argument and generate AI response using GA and AS
        
        Args:
            debate_id: The debate ID
            argument: The user's argument text
            
        Returns:
            Dict with response data, scores, and feedback
        """
        debate = self.debates.get(debate_id)
        if not debate:
            raise ValueError("Debate not found")

        if debate["current_round"] > debate["max_rounds"]:
            raise ValueError("Debate has ended")

        current_round = debate["current_round"]
        
        # 1. Extract argument features and detect debate move using AS
        argument_features = self.move_predictor.extract_features(argument)
        detected_move = self._detect_debate_move(argument) 
        
        # Update AS model with the detected move
        self.move_predictor.add_move("Current State", detected_move)
        debate["detected_moves"].append(detected_move)
        
        # 2. Evaluate the user's argument (using LLaMA for evaluation)
        user_score, user_feedback = self._parse_evaluation(
            evaluate_argument(argument, debate["topic"])
        )
        
        # 3. Get predicted next move and counter from AS
        next_move_prediction = self.move_predictor.predict_best_move(
            "Current State", depth=2, features=argument_features
        )
        counter_move = self.move_predictor.get_counter_move(next_move_prediction.move_type)
        
        # Store prediction and counter
        debate["as_predictions"].append(next_move_prediction.move_type)
        debate["counter_moves"].append(counter_move)
        
        # 4. Get the current best GA strategy 
        best_strategy = self.strategy_evolver.get_best_strategy(debate["ga_fitness"])
        debate["best_strategy"] = best_strategy
        
        # 5. Convert strategy to dict for LLM integration
        strategy_dict = self._strategy_to_dict(best_strategy)
        
        # 6. Improve strategy with AS predictions and counter
        strategy_dict["predicted_user_move"] = next_move_prediction.move_type
        strategy_dict["predicted_confidence"] = next_move_prediction.score
        strategy_dict["recommended_counter"] = counter_move
        
        # Add sequence prediction if available
        move_sequence = self.move_predictor.predict_move_sequence("Current State", 2)
        if move_sequence:
            strategy_dict["predicted_sequence"] = [m.move_type for m in move_sequence]
        
        # 7. Generate AI's response using the integrated GA+AS strategy with Gemma
        ai_argument = get_ai_opponent_response(
            debate["topic"], argument, debate["ai_position"], strategy_dict
        )
        
        # 8. Evaluate AI's response with LLaMA
        ai_score, ai_feedback = self._parse_evaluation(
            evaluate_argument(ai_argument, debate["topic"])
        )
        
        # 9. Update GA fitness based on AI's performance
        # Apply current score with random variation to create evolutionary pressure
        for i in range(len(debate["ga_fitness"])):
            # Random variation to avoid local optima
            variation = random.uniform(-0.1, 0.1)
            new_fitness = ai_score + variation
            
            # Ensure fitness stays in valid range
            new_fitness = max(0.1, min(1.0, new_fitness))
            debate["ga_fitness"][i] = new_fitness
        
        # 10. Run multiple generations to accelerate evolution
        for _ in range(self.generations_per_round):
            self.strategy_evolver.evolve(debate["ga_fitness"])
        
        # 11. Get post-evolution best strategy
        evolved_strategy = self.strategy_evolver.get_best_strategy(debate["ga_fitness"])
        evolved_dict = self._strategy_to_dict(evolved_strategy)
        
        # 12. Format GA feedback for user
        ga_feedback = self._format_strategy_feedback(evolved_strategy, detected_move)
        
        # 13. Get educational feedback using Phi model (focusing on educational value)
        assistant_feedback = get_debate_assistant_response(
            debate["topic"], argument, debate["user_position"], evolved_dict, 
            {"detected_move": detected_move, "predicted_counter": counter_move}
        )
        
        # 14. Generate strategy-informed suggestions using LLaMA
        llm_suggestions = generate_llm_suggestions(
            debate["topic"], debate["user_position"], evolved_dict
        )
        
        # 15. Get improvement suggestions from Phi (focusing on educational value)
        improvement_suggestions = suggest_improvements(
            debate["topic"], argument, debate["user_position"], evolved_dict,
            {"detected_move": detected_move, "predicted_counter": counter_move}
        )
        
        # Combine improvement suggestions with LLM suggestions if we need more
        if len(llm_suggestions) < 3 and improvement_suggestions:
            for suggestion in improvement_suggestions:
                if len(llm_suggestions) < 3 and suggestion not in llm_suggestions:
                    llm_suggestions.append(suggestion)
        
        # 16. Update debate state
        debate["scores"]["user"] += user_score
        debate["scores"]["ai"] += ai_score
        debate["arguments"]["user"][current_round] = argument
        debate["arguments"]["ai"][current_round] = ai_argument
        debate["evaluations"]["user"][current_round] = user_feedback
        debate["evaluations"]["ai"][current_round] = ai_feedback
        debate["move_history"].append(detected_move)
        
        # Record evolution for history
        debate["evolution_history"].append({
            "round": current_round,
            "pre_strategy": strategy_dict,
            "post_strategy": evolved_dict,
            "score": ai_score
        })
        
        # 17. Increment round
        debate["current_round"] += 1
        
        # 18. Get AS model stats
        as_stats = self.move_predictor.get_model_stats()
        
        # 19. Format response
        return {
            "score": user_score,
            "ai_score": ai_score,
            "ai_argument": ai_argument,
            "ga_feedback": ga_feedback,
            "as_prediction": f"Predicted move: {next_move_prediction.move_type}, Confidence: {next_move_prediction.score:.2f}",
            "llm_suggestions": llm_suggestions,
            "current_round": debate["current_round"],
            "max_rounds": debate["max_rounds"],
            "evaluation_feedback": user_feedback,
            "debate_assistant_feedback": assistant_feedback,
            "arguments": {
                "round_" + str(current_round): {
                    "user": argument,
                    "ai": ai_argument
                }
            },
            "evolution_stats": {
                "pre_strategy": strategy_dict,
                "post_strategy": evolved_dict
            },
            "as_model_stats": as_stats
        }

    def _detect_debate_move(self, argument: str) -> str:
        """Detect the debate move type from argument text"""
        text = argument.lower()
        
        # Feature detection for move classification
        move_scores = {}
        
        # Evidence-based scoring
        evidence_terms = ["evidence", "research", "study", "data", "according to", "statistics"]
        move_scores["present_evidence"] = sum(text.count(term) for term in evidence_terms) / len(evidence_terms)
        
        # Emotional appeal scoring
        emotion_terms = ["feel", "emotional", "heart", "suffering", "hope", "fear", "devastating"]
        move_scores["appeal_to_emotion"] = sum(text.count(term) for term in emotion_terms) / len(emotion_terms)
        
        # Logical reasoning scoring
        logic_terms = ["therefore", "thus", "consequently", "logically", "reason", "conclusion"]
        move_scores["use_logical_reasoning"] = sum(text.count(term) for term in logic_terms) / len(logic_terms)
        
        # Counterargument scoring
        counter_terms = ["however", "although", "despite", "nevertheless", "while", "critics"]
        move_scores["address_counterarguments"] = sum(text.count(term) for term in counter_terms) / len(counter_terms)
        
        # Example scoring
        example_terms = ["example", "instance", "case", "such as", "for instance"]
        move_scores["provide_examples"] = sum(text.count(term) for term in example_terms) / len(example_terms)
        
        # Analogy scoring
        analogy_terms = ["like", "as", "similar", "compared to", "analogy"]
        move_scores["use_analogy"] = sum(text.count(term) for term in analogy_terms) / len(analogy_terms)
        
        # Consequence highlighting scoring
        consequence_terms = ["result", "impact", "effect", "consequence", "lead to"]
        move_scores["highlight_consequences"] = sum(text.count(term) for term in consequence_terms) / len(consequence_terms)
        
        # Rhetorical question scoring
        move_scores["ask_rhetorical_questions"] = text.count('?') / max(1, len(text.split('.')))
        
        # Return move with highest score or default
        if move_scores:
            best_move = max(move_scores.items(), key=lambda x: x[1])
            if best_move[1] > 0:
                return best_move[0]
        
        return "general_argument"

    def _strategy_to_dict(self, strategy: Strategy) -> Dict:
        """Convert a Strategy object to dictionary for LLM integration"""
        if not strategy:
            return {'ethos': 0.33, 'pathos': 0.33, 'logos': 0.34, 'tactics': []}
        
        strategy_dict = {}
        
        # Get rhetorical balance
        genes = strategy.genes
        strategy_dict['ethos'] = genes.get('ethos', 0.33)
        strategy_dict['pathos'] = genes.get('pathos', 0.33) 
        strategy_dict['logos'] = genes.get('logos', 0.34)
        
        # Get tactics
        strategy_dict['tactics'] = strategy.get_dominant_tactics(3)
        
        return strategy_dict

    def _format_strategy_feedback(self, strategy: Strategy, detected_move: str) -> str:
        """Format GA strategy as user-friendly feedback"""
        if not strategy:
            return "Focus on balancing logical arguments with emotional appeals."
        
        # Extract rhetorical balance
        ethos_pct = strategy.genes.get('ethos', 0.33) * 100
        pathos_pct = strategy.genes.get('pathos', 0.33) * 100
        logos_pct = strategy.genes.get('logos', 0.34) * 100
        
        # Format core feedback
        feedback = f"Focus on {ethos_pct:.1f}% ethos (credibility), {pathos_pct:.1f}% pathos (emotional appeal), and {logos_pct:.1f}% logos (logical reasoning)."
        
        # Add tactics
        tactics = strategy.get_dominant_tactics(3)
        if tactics:
            # Convert tactics to readable format
            readable_tactics = []
            for tactic in tactics:
                readable = tactic.replace('_', ' ')
                readable_tactics.append(readable)
            
            feedback += f" Try {', '.join(readable_tactics)}."
        
        # Add counter advice based on detected move
        if detected_move == "present_evidence":
            feedback += " Consider balancing facts with emotional context."
        elif detected_move == "appeal_to_emotion":
            feedback += " Add more concrete evidence to strengthen your emotional points."
        elif detected_move == "use_logical_reasoning":
            feedback += " Make your logic more relatable with examples."
        
        return feedback

    def _parse_evaluation(self, evaluation: str) -> Tuple[float, str]:
        """Parse evaluation text to extract score and feedback"""
        try:
            # Extract score (/10) from evaluation
            score_match = re.search(r'(\d+(\.\d+)?)\s*(/|\s*out of\s*)\s*10', evaluation)
            if score_match:
                score = float(score_match.group(1))
            else:
                raise ValueError("Score not found in evaluation")

            # Normalize score to 0-1 range
            normalized_score = score / 10
            
            # Extract feedback (text after score)
            feedback = re.sub(r'^.*?(/|\s*out of\s*)\s*10', '', evaluation, flags=re.DOTALL).strip()

            return normalized_score, feedback
        except Exception as e:
            # Fallback for parsing errors
            print(f"Error parsing evaluation: {e}")
            return 0.5, "Error in evaluation"

    def generate_llm_suggestions(self, debate_id: str) -> List[str]:
        """Generate strategic argument suggestions for the user using LLaMA"""
        debate = self.debates.get(debate_id)
        if not debate:
            raise ValueError("Debate not found")
        
        # Get latest best strategy
        best_strategy = debate.get("best_strategy")
        strategy_dict = self._strategy_to_dict(best_strategy) if best_strategy else None
        
        # Generate suggestions with strategic influence using LLaMA
        return generate_llm_suggestions(debate["topic"], debate["user_position"], strategy_dict)

    def get_debate_state(self, debate_id: str) -> Dict:
        """Get current state of a debate with GA and AS insights"""
        debate = self.debates.get(debate_id)
        if not debate:
            raise ValueError("Debate not found")
        
        # Get current best strategy
        best_strategy = debate.get("best_strategy")
        strategy_dict = self._strategy_to_dict(best_strategy) if best_strategy else None
        
        # Format GA strategy for display
        if best_strategy:
            ethos = best_strategy.genes.get('ethos', 0.33) * 100
            pathos = best_strategy.genes.get('pathos', 0.33) * 100
            logos = best_strategy.genes.get('logos', 0.34) * 100
            
            ga_strategy = f"Focus on {ethos:.1f}% ethos, {pathos:.1f}% pathos, {logos:.1f}% logos"
            
            tactics = best_strategy.get_dominant_tactics(3)
            if tactics:
                tactics_str = ", ".join(t.replace('_', ' ') for t in tactics)
                ga_strategy += f". Recommended tactics: {tactics_str}"
        else:
            ga_strategy = "Developing strategy..."
        
        # Get latest AS prediction
        next_move = self.move_predictor.predict_best_move("Current State", 2)
        
        # Get move sequence prediction
        move_sequence = self.move_predictor.predict_move_sequence("Current State", 2)
        sequence_str = " → ".join([m.move_type for m in move_sequence]) if move_sequence else ""
        
        # Get AS stats
        as_stats = self.move_predictor.get_model_stats()
        
        # Format arguments for current state
        arguments = {}
        for round_num in range(1, debate["current_round"]):
            arguments[f"round_{round_num}"] = {
                "user": debate["arguments"]["user"].get(round_num, ""),
                "ai": debate["arguments"]["ai"].get(round_num, "")
            }
        
        # Generate suggestions using LLaMA
        llm_suggestions = self.generate_llm_suggestions(debate_id)
        
        return {
            "topic": debate["topic"],
            "current_round": debate["current_round"],
            "max_rounds": debate["max_rounds"],
            "scores": debate["scores"],
            "arguments": arguments,
            "user_position": debate["user_position"],
            "ai_position": debate["ai_position"],
            "current_turn": "user" if debate["current_round"] <= debate["max_rounds"] else "completed",
            "status": "in_progress" if debate["current_round"] <= debate["max_rounds"] else "completed",
            "llm_suggestions": llm_suggestions,
            "ga_strategy": ga_strategy,
            "as_prediction": next_move.move_type,
            "as_prediction_confidence": next_move.score,
            "as_predicted_sequence": sequence_str,
            "evolution_history": debate["evolution_history"][-1] if debate["evolution_history"] else None,
            "as_model_stats": as_stats
        }