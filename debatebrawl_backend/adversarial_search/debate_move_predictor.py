from typing import List, Tuple, Dict
import random

class Move:
    def __init__(self, move: str, score: float):
        self.move = move
        self.score = score

class DebateMovePredictor:
    def __init__(self, initial_state: str):
        self.state = initial_state
        self.move_history: List[Tuple[str, str]] = []
        self.state_transition_probabilities: Dict[str, Dict[str, int]] = {}
        self.debate_tactics = [
            "present evidence", "appeal to emotion", "use logical reasoning",
            "address counterarguments", "provide examples", "use analogy",
            "cite expert opinion", "highlight consequences", "reframe the issue",
            "ask rhetorical questions"
        ]

    def add_move(self, state: str, move: str):
        self.move_history.append((state, move))
        self.update_transition_probabilities(state, move)

    def update_transition_probabilities(self, state: str, move: str):
        if state not in self.state_transition_probabilities:
            self.state_transition_probabilities[state] = {}
        if move not in self.state_transition_probabilities[state]:
            self.state_transition_probabilities[state][move] = 0
        self.state_transition_probabilities[state][move] += 1

    def predict_best_move(self, current_state: str, depth: int) -> Move:
        if current_state not in self.state_transition_probabilities:
            return Move(random.choice(self.debate_tactics), 0.5)

        moves = self.state_transition_probabilities[current_state]
        total_moves = sum(moves.values())
        probabilities = {move: count / total_moves for move, count in moves.items()}

        best_move = max(probabilities, key=probabilities.get)
        confidence = probabilities[best_move]

        return Move(best_move, confidence)

    def calculate_similarity(self, state1: str, state2: str) -> float:
        words1 = set(state1.lower().split())
        words2 = set(state2.lower().split())
        common_words = words1.intersection(words2)
        return len(common_words) / max(len(words1), len(words2))

    def find_similar_states(self, current_state: str, similarity_threshold: float = 0.5) -> List[str]:
        similar_states = []
        for state in self.state_transition_probabilities.keys():
            similarity = self.calculate_similarity(state, current_state)
            if similarity > similarity_threshold:
                similar_states.append(state)
        return similar_states

    def predict_move_sequence(self, current_state: str, depth: int) -> List[Move]:
        move_sequence = []
        state = current_state

        for _ in range(depth):
            move = self.predict_best_move(state, 1)
            move_sequence.append(move)
            state = move.move  # Assume the move becomes the new state

        return move_sequence

    def get_counter_move(self, predicted_move: str) -> str:
        counter_moves = {
            "present evidence": "question evidence validity",
            "appeal to emotion": "focus on facts and logic",
            "use logical reasoning": "find logical fallacies",
            "address counterarguments": "introduce new arguments",
            "provide examples": "show exceptions to examples",
            "use analogy": "point out analogy limitations",
            "cite expert opinion": "cite conflicting expert opinions",
            "highlight consequences": "show alternative outcomes",
            "reframe the issue": "reinforce original framing",
            "ask rhetorical questions": "provide direct answers"
        }
        return counter_moves.get(predicted_move, random.choice(list(counter_moves.values())))