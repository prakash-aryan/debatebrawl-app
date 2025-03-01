import random
import math
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter

class Move:
    """Represents a debate move with type and confidence score"""
    def __init__(self, move_type: str, score: float, features: Optional[Dict] = None):
        self.move_type = move_type
        self.score = score
        self.features = features or {}
    
    def __str__(self):
        return f"{self.move_type} (confidence: {self.score:.2f})"

class Node:
    """Tree node for Monte Carlo Tree Search"""
    def __init__(self, state, move=None, parent=None):
        self.state = state
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
    
    def add_child(self, child_state, move):
        child = Node(child_state, move, self)
        self.children.append(child)
        return child
    
    def update(self, result):
        self.visits += 1
        self.value += result
    
    def ucb_score(self, exploration_weight=1.0):
        if self.visits == 0:
            return float('inf')
        exploitation = self.value / self.visits
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def best_child(self, exploration_weight=1.0):
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.ucb_score(exploration_weight))

class DebateMovePredictor:
    """Enhanced Adversarial Search component for debate move prediction"""
    
    def __init__(self, initial_state: str):
        self.initial_state = initial_state
        self.move_history = []
        self.state_transition_matrix = {}
        
        # Debate tactics/moves
        self.debate_tactics = [
            "present_evidence", "appeal_to_emotion", "use_logical_reasoning",
            "address_counterarguments", "provide_examples", "use_analogy",
            "cite_expert_opinion", "highlight_consequences", "reframe_issue",
            "ask_rhetorical_questions", "establish_credibility", "tell_story",
            "concede_points", "build_common_ground", "general_argument"
        ]
        
        # Initialize transition matrix
        self._initialize_transition_matrix()
        
        # Improved feature detection
        self.keyword_features = self._initialize_enhanced_keywords()
        self.phrase_patterns = self._initialize_phrase_patterns()
        
        # Improved pattern recognition
        self.sequence_patterns = defaultdict(Counter)
        self.max_sequence_length = 3
        self.recency_weight = 2.0  # Weight recent patterns more heavily
        
        # Strategy effectiveness tracking
        self.move_effectiveness = {tactic: 0.5 for tactic in self.debate_tactics}
        
        # Counter-move effectiveness matrix
        self.counter_strategy = self._initialize_counter_matrix()
        
        # Dynamic exploration for MCTS
        self.base_exploration_factor = 0.7
        self.exploration_adjustment = 0.0
        self.confidence_history = []

    def _initialize_transition_matrix(self):
        """Initialize the Markov transition matrix for moves"""
        matrix = {}
        for move1 in self.debate_tactics:
            matrix[move1] = {}
            for move2 in self.debate_tactics:
                # Equal initial probability
                matrix[move1][move2] = 1.0 / len(self.debate_tactics)
        return matrix

    def _initialize_counter_matrix(self):
        """Initialize the counter-move effectiveness matrix with improved counters"""
        counter_matrix = {}
        
        # Define effective counter relationships - more precise counters
        counter_relationships = {
            "present_evidence": ["address_counterarguments", "reframe_issue"],
            "appeal_to_emotion": ["present_evidence", "use_logical_reasoning"],
            "use_logical_reasoning": ["appeal_to_emotion", "ask_rhetorical_questions"],
            "address_counterarguments": ["present_evidence", "highlight_consequences"],
            "provide_examples": ["reframe_issue", "address_counterarguments"],
            "use_analogy": ["use_logical_reasoning", "present_evidence"],
            "cite_expert_opinion": ["cite_expert_opinion", "establish_credibility"],
            "highlight_consequences": ["address_counterarguments", "reframe_issue"],
            "reframe_issue": ["highlight_consequences", "use_logical_reasoning"],
            "ask_rhetorical_questions": ["present_evidence", "use_logical_reasoning"],
            "establish_credibility": ["address_counterarguments", "use_logical_reasoning"],
            "tell_story": ["use_logical_reasoning", "present_evidence"],
            "concede_points": ["present_evidence", "highlight_consequences"],
            "build_common_ground": ["present_evidence", "highlight_consequences"],
            "general_argument": ["present_evidence", "ask_rhetorical_questions"]
        }
        
        # Initialize matrix with neutral effectiveness
        for move1 in self.debate_tactics:
            counter_matrix[move1] = {move2: 0.5 for move2 in self.debate_tactics}
            
            # Set higher effectiveness for known counters
            if move1 in counter_relationships:
                for counter_move in counter_relationships[move1]:
                    counter_matrix[move1][counter_move] = 0.8
        
        return counter_matrix

    def _initialize_enhanced_keywords(self):
        """Initialize expanded keyword sets for more accurate feature detection"""
        return {
            "present_evidence": [
                "evidence", "research", "study", "data", "statistics", "shows", "according", 
                "survey", "experiment", "finding", "measurement", "report", "analysis",
                "concluded", "demonstrated", "revealed", "confirmed", "verified", 
                "empirical", "quantitative", "qualitative", "peer-reviewed"
            ],
            "appeal_to_emotion": [
                "feel", "emotional", "heart", "suffering", "hope", "fear", "pain", 
                "devastated", "inspiring", "tragic", "alarming", "disturbing", "moving",
                "touching", "horrifying", "shocking", "distressing", "concerning",
                "worry", "anxious", "frightening", "uplifting", "depressing"
            ],
            "use_logical_reasoning": [
                "therefore", "thus", "consequently", "logically", "it follows", "reasoning",
                "infer", "deduce", "conclude", "implies", "premise", "syllogism",
                "if-then", "causation", "correlation", "necessarily", "inevitably",
                "rational", "analysis", "demonstrates", "proves", "indicates"
            ],
            "address_counterarguments": [
                "however", "although", "despite", "nevertheless", "opponents", "critics",
                "some argue", "contrary", "challenge", "objection", "criticism",
                "counterpoint", "alternative view", "opposing", "refute", "rebut",
                "concede", "acknowledge", "granted", "admittedly", "while"
            ],
            "provide_examples": [
                "example", "instance", "such as", "consider", "case", "illustration",
                "demonstrates", "exemplifies", "illustrates", "case study", "specifically",
                "for instance", "to illustrate", "namely", "particularly", "as seen in"
            ],
            "use_analogy": [
                "like", "similar to", "just as", "compared to", "analogy", "metaphor",
                "resembles", "parallels", "mirrors", "equivalent to", "akin to",
                "comparable", "corresponds to", "in the same way", "similarly"
            ],
            "cite_expert_opinion": [
                "expert", "authority", "professor", "specialist", "researcher", "according to",
                "scholar", "scientist", "professional", "doctorate", "peer-reviewed",
                "renowned", "acclaimed", "leading", "prominent", "distinguished"
            ],
            "highlight_consequences": [
                "result", "outcome", "impact", "effect", "consequence", "lead to",
                "causes", "produces", "generates", "creates", "brings about",
                "implications", "ramifications", "aftermath", "repercussions",
                "ultimately", "eventually", "in the long run", "inevitably"
            ],
            "reframe_issue": [
                "perspective", "context", "view", "framework", "lens", "consider",
                "reframe", "reconsider", "rethink", "recontextualize", "redefine",
                "shift focus", "alternative", "another way", "instead", "rather"
            ],
            "ask_rhetorical_questions": [
                "?", "how can", "why would", "what if", "consider this", "isn't it",
                "wouldn't you", "shouldn't we", "can we really", "who could",
                "where does", "when has", "how might", "does this not", "is it not"
            ],
            "establish_credibility": [
                "experience", "background", "qualification", "trained", "certified",
                "credential", "expertise", "specialized", "practiced", "professional",
                "qualified", "authorized", "competent", "proficient", "skilled"
            ],
            "tell_story": [
                "story", "once", "imagine", "picture this", "scene", "narrative",
                "anecdote", "account", "tale", "incident", "episode", "scenario",
                "situation", "experience", "recollection", "memory", "recall"
            ],
            "concede_points": [
                "agree", "concede", "granted", "admittedly", "fair point", "valid",
                "legitimate", "reasonable", "justified", "understandable", "acceptable",
                "acknowledge", "recognize", "admit", "true", "correct", "right"
            ],
            "build_common_ground": [
                "common", "shared", "agree", "both", "together", "mutual",
                "consensus", "middle ground", "compromise", "understanding",
                "cooperation", "collaboration", "joint", "collective", "unified"
            ]
        }

    def _initialize_phrase_patterns(self):
        """Initialize multi-word phrases for better move detection"""
        return {
            "present_evidence": [
                "according to research", "studies show", "data indicates", 
                "evidence suggests", "research demonstrates", "statistics reveal",
                "the findings confirm", "analysis shows", "survey results"
            ],
            "appeal_to_emotion": [
                "think about how", "imagine the pain", "consider the suffering",
                "feel the impact", "heart-wrenching", "deeply concerning",
                "emotionally devastating", "profoundly disturbing"
            ],
            "use_logical_reasoning": [
                "it follows that", "we can conclude", "logically speaking",
                "this implies that", "the conclusion is", "reasoning suggests",
                "this demonstrates that", "it stands to reason"
            ],
            "address_counterarguments": [
                "critics might argue", "some may object", "opponents claim",
                "contrary to popular belief", "despite objections", "although some believe",
                "while others maintain", "it could be argued"
            ],
            "reframe_issue": [
                "from another perspective", "looking at it differently",
                "consider instead", "reframing the issue", "shift our focus",
                "alternative viewpoint", "different approach", "another way to see this"
            ]
        }

    def extract_features(self, argument: str) -> Dict[str, float]:
        """Extract linguistic features from argument text with enhanced detection"""
        features = {tactic: 0.0 for tactic in self.debate_tactics}
        text = argument.lower()
        
        # Check for keyword presence with weighted detection
        for tactic, keywords in self.keyword_features.items():
            # Count keyword occurrences with word boundaries
            matches = sum(text.count(f" {keyword} ") for keyword in keywords)
            # Add partial matches
            partial_matches = sum(text.count(keyword) for keyword in keywords)
            # Calculate normalized score
            feature_score = (matches + partial_matches * 0.5) / (len(keywords) * 0.3)
            features[tactic] = min(1.0, feature_score)
        
        # Check for phrase patterns - stronger indicators
        for tactic, phrases in self.phrase_patterns.items():
            matches = sum(text.count(phrase) for phrase in phrases)
            if matches > 0:
                # Phrases are stronger indicators, so they add more weight
                features[tactic] = min(1.0, features[tactic] + matches * 0.4)
        
        # Special cases for certain move types
        # Count questions for rhetorical questions
        question_count = text.count('?')
        if question_count > 0:
            features["ask_rhetorical_questions"] = min(1.0, question_count / max(1, len(text.split('.'))))
        
        # Check for narrative structure indicators for storytelling
        if any(marker in text for marker in ["once", "when i", "i remember", "years ago"]):
            features["tell_story"] = max(features["tell_story"], 0.7)
        
        # Detect concession language
        if any(concession in text for concession in ["i agree", "fair point", "that's true", "i concede"]):
            features["concede_points"] = max(features["concede_points"], 0.8)
        
        # Topic-specific term detection could be added here
        
        return features

    def add_move(self, state: str, move: str, effectiveness: float = None):
        """Add a move to history and update transition probabilities with recency weighting"""
        # Record the move
        self.move_history.append((state, move))
        
        # Update transition probabilities if we have previous moves
        if len(self.move_history) > 1:
            prev_state, prev_move = self.move_history[-2]
            
            # Initialize transition matrix for this move if needed
            if prev_move not in self.state_transition_matrix:
                self.state_transition_matrix[prev_move] = {}
            
            # Increment count for this transition with recency weighting
            if move not in self.state_transition_matrix[prev_move]:
                self.state_transition_matrix[prev_move][move] = 0
            
            # Recent observations are weighted more heavily
            recency_weight = min(2.0, 1.0 + (0.1 * len(self.move_history)))
            self.state_transition_matrix[prev_move][move] += recency_weight
            
            # Normalize to maintain probability distribution
            total = sum(self.state_transition_matrix[prev_move].values())
            for next_move in self.state_transition_matrix[prev_move]:
                self.state_transition_matrix[prev_move][next_move] /= total
        
        # Update sequence patterns with recency weighting
        if len(self.move_history) >= 2:
            for seq_len in range(2, min(self.max_sequence_length + 1, len(self.move_history))):
                # Create sequence from previous moves
                sequence = tuple(m for _, m in self.move_history[-seq_len:-1])
                
                # Weight recent patterns more heavily
                recency_factor = 1.0
                if len(self.move_history) > 5:
                    # Increase weight for more recent patterns
                    recency_factor = self.recency_weight
                
                self.sequence_patterns[sequence][move] += recency_factor
        
        # Update effectiveness if provided
        if effectiveness is not None:
            # Exponential moving average with higher weight to recent outcomes
            self.move_effectiveness[move] = 0.7 * self.move_effectiveness.get(move, 0.5) + 0.3 * effectiveness
        
        # Update confidence history for adaptive exploration
        if len(self.move_history) > 1:
            prev_confidence = self.confidence_history[-1] if self.confidence_history else 0.5
            current_confidence = self.state_transition_matrix.get(self.move_history[-2][1], {}).get(move, 0.5)
            self.confidence_history.append(current_confidence)
            
            # Adjust exploration factor based on confidence trend
            if len(self.confidence_history) >= 3:
                # Check if confidence is improving
                recent_trend = self.confidence_history[-1] - self.confidence_history[-3]
                if recent_trend > 0.1:
                    # Confidence improving, reduce exploration to exploit known patterns
                    self.exploration_adjustment = max(-0.3, self.exploration_adjustment - 0.1)
                elif recent_trend < -0.1:
                    # Confidence decreasing, increase exploration
                    self.exploration_adjustment = min(0.3, self.exploration_adjustment + 0.1)

    def update_counter_effectiveness(self, move: str, counter: str, effectiveness: float):
        """Update the effectiveness of a counter-strategy with higher weight to recent results"""
        if move in self.counter_strategy and counter in self.counter_strategy[move]:
            # Update with higher weight to recent results (0.3 instead of 0.2)
            current = self.counter_strategy[move][counter]
            self.counter_strategy[move][counter] = 0.7 * current + 0.3 * effectiveness

    def predict_best_move(self, current_state: str, depth: int = 2, features: Dict = None) -> Move:
        """Predict the best next move using enhanced pattern matching and MCTS"""
        # Extract features if not provided
        arg_features = features or {}
        
        # If we have move history, check for strong patterns first
        if len(self.move_history) > 0:
            # Check if we have sequence patterns, prioritizing longer sequences
            for seq_len in range(min(self.max_sequence_length, len(self.move_history)), 0, -1):
                # Get the sequence of previous moves
                sequence = tuple(m for _, m in self.move_history[-seq_len:])
                
                if sequence in self.sequence_patterns and self.sequence_patterns[sequence]:
                    # Get most common next move after this sequence
                    most_common = self.sequence_patterns[sequence].most_common(1)
                    if most_common:
                        move_type, count = most_common[0]
                        total = sum(self.sequence_patterns[sequence].values())
                        confidence = count / total
                        if confidence > 0.7:  # Strong pattern threshold
                            return Move(move_type, min(confidence, 0.9), arg_features)
            
            # Check feature-based prediction for newer move types
            if arg_features:
                # Find move types with high feature scores
                strong_features = [(move, score) for move, score in arg_features.items() if score > 0.7]
                if strong_features:
                    strongest = max(strong_features, key=lambda x: x[1])
                    # If we have a very strong feature match, use it
                    if strongest[1] > 0.8:
                        return Move(strongest[0], strongest[1], arg_features)
        
        # For deeper analysis, use Monte Carlo Tree Search with adaptive exploration
        if depth >= 2 and len(self.move_history) > 0:
            adaptive_exploration = self.base_exploration_factor + self.exploration_adjustment
            mcts_move = self._mcts_search(current_state, depth, adaptive_exploration)
            if mcts_move and mcts_move.score > 0.6:
                return mcts_move
        
        # Fall back to transition probabilities if possible
        if len(self.move_history) > 0:
            prev_move = self.move_history[-1][1]
            if prev_move in self.state_transition_matrix:
                # Get probabilities of next moves
                next_move_probs = self.state_transition_matrix[prev_move]
                
                # Find highest probability move
                if next_move_probs:
                    best_move = max(next_move_probs.items(), key=lambda x: x[1])
                    return Move(best_move[0], best_move[1], arg_features)
        
        # If no history or prediction, return general argument with medium confidence
        return Move("general_argument", 0.5, arg_features)

    def _mcts_search(self, current_state: str, depth: int, exploration_factor: float = 0.7) -> Optional[Move]:
        """Perform Monte Carlo Tree Search to find optimal next move with improved simulation"""
        # Create root node
        root = Node(current_state)
        
        # Run MCTS for more simulations if we have more history
        simulation_count = 100 + min(100, len(self.move_history) * 10)
        
        # Run MCTS for fixed number of iterations
        for _ in range(simulation_count):
            # Selection phase - traverse tree using UCB
            node = root
            while node.children and node.visits > 0:
                node = node.best_child(exploration_factor)
            
            # Expansion phase - add new nodes if not leaf
            if node.visits > 0 and len(node.children) < len(self.debate_tactics):
                # Find unexplored moves
                explored_moves = {child.move for child in node.children}
                unexplored = [m for m in self.debate_tactics if m not in explored_moves]
                
                if unexplored:
                    # Choose a move based on transition probabilities if available
                    weights = None
                    if len(self.move_history) > 0:
                        prev_move = self.move_history[-1][1] if self.move_history else None
                        if prev_move in self.state_transition_matrix:
                            weights = [self.state_transition_matrix[prev_move].get(m, 0.1) for m in unexplored]
                    
                    # Select move with weights or randomly if no weights
                    new_move = random.choices(unexplored, weights=weights, k=1)[0] if weights else random.choice(unexplored)
                    new_state = f"{node.state} -> {new_move}"
                    node = node.add_child(new_state, new_move)
            
            # Simulation phase - improved simulation with realistic move patterns
            simulation_score = self._improved_simulation(node, depth)
            
            # Backpropagation phase - update values up the tree
            while node:
                node.update(simulation_score)
                node = node.parent
        
        # Choose best move from root children
        if not root.children:
            return None
        
        best_child = max(root.children, key=lambda c: c.value / c.visits if c.visits > 0 else 0)
        return Move(best_child.move, best_child.value / best_child.visits)

    def _improved_simulation(self, node: Node, depth: int) -> float:
        """Improved debate simulation using learned patterns and counter-strategies"""
        if depth <= 0:
            return 0.5  # Neutral score at max depth
        
        current_move = node.move
        if not current_move:
            # If no move, choose one based on move effectiveness
            moves = list(self.move_effectiveness.keys())
            weights = [self.move_effectiveness[m] for m in moves]
            current_move = random.choices(moves, weights=weights, k=1)[0]
        
        # Choose counter move based on learned counter effectiveness
        if current_move in self.counter_strategy:
            counter_moves = list(self.counter_strategy[current_move].keys())
            counter_weights = [self.counter_strategy[current_move][m] for m in counter_moves]
            
            # Take the top 3 counter moves for more focused simulation
            top_counters = sorted(zip(counter_moves, counter_weights), key=lambda x: x[1], reverse=True)[:3]
            if top_counters:
                counter_moves = [c[0] for c in top_counters]
                counter_weights = [c[1] for c in top_counters]
                counter_move = random.choices(counter_moves, weights=counter_weights, k=1)[0]
            else:
                counter_move = random.choice(self.debate_tactics)
        else:
            counter_move = random.choice(self.debate_tactics)
        
        # Get effectiveness of counter
        effectiveness = self.counter_strategy.get(current_move, {}).get(counter_move, 0.5)
        
        # Use learned sequence patterns for more accurate simulation
        seq_effectiveness = 0.5
        for seq_len in range(2, min(self.max_sequence_length + 1, len(self.move_history) + 1)):
            if len(self.move_history) >= seq_len - 1:
                # Get recent move sequence
                recent_moves = [m for _, m in self.move_history[-(seq_len-1):]]
                if recent_moves and recent_moves[-1] == current_move:
                    # Check if this sequence has a learned pattern
                    sequence = tuple(recent_moves)
                    if sequence in self.sequence_patterns and counter_move in self.sequence_patterns[sequence]:
                        # Use the pattern's probability as effectiveness estimate
                        total = sum(self.sequence_patterns[sequence].values())
                        seq_effectiveness = self.sequence_patterns[sequence][counter_move] / total
                        break
        
        # Combine counter-strategy effectiveness with sequence pattern effectiveness
        combined_effectiveness = effectiveness * 0.6 + seq_effectiveness * 0.4
        
        # Recursive simulation with decaying importance
        return 0.7 * combined_effectiveness + 0.3 * self._improved_simulation(
            Node(f"{node.state} -> {counter_move}", counter_move), depth - 1)

    def get_counter_move(self, predicted_move: str) -> str:
        """Get the most effective counter to a predicted move"""
        if predicted_move not in self.counter_strategy:
            return random.choice(self.debate_tactics)
        
        # Get counter effectiveness scores
        counter_scores = self.counter_strategy[predicted_move]
        
        # Select the most effective counter - with weighted randomness for variety
        top_counters = sorted(counter_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        if not top_counters:
            return random.choice(self.debate_tactics)
        
        moves = [m[0] for m in top_counters]
        weights = [m[1] for m in top_counters]
        return random.choices(moves, weights=weights, k=1)[0]

    def predict_move_sequence(self, current_state: str, depth: int) -> List[Move]:
        """Predict a sequence of future moves with improved sequence modeling"""
        moves = []
        state = current_state
        
        # Use more sophisticated sequence prediction with MCTS
        for i in range(depth):
            # Use deeper search for first move, shallower for later moves
            search_depth = max(1, 3 - i)
            next_move = self.predict_best_move(state, search_depth)
            moves.append(next_move)
            state = f"{state} -> {next_move.move_type}"
            
            # After adding a move, immediately consider counter-moves
            if i < depth - 1:
                # Consider the counter to this move for the next prediction
                counter_move = self.get_counter_move(next_move.move_type)
                state = f"{state} -> (counter: {counter_move})"
        
        return moves

    def get_model_stats(self) -> Dict:
        """Get enhanced statistics about the model's performance and state"""
        # Calculate entropy of transition matrix as a measure of predictability
        entropy = 0.0
        for from_move, transitions in self.state_transition_matrix.items():
            for to_move, prob in transitions.items():
                if prob > 0:
                    entropy -= prob * math.log(prob)
        
        # Extract top patterns
        top_patterns = {}
        for seq, counters in self.sequence_patterns.items():
            if counters:
                top_move, count = counters.most_common(1)[0]
                total = sum(counters.values())
                top_patterns[" → ".join(seq)] = (top_move, count/total)
        
        return {
            "transition_entropy": entropy,
            "moves_recorded": len(self.move_history),
            "top_move_patterns": dict(sorted(top_patterns.items(), 
                                           key=lambda x: x[1][1], 
                                           reverse=True)[:5]),
            "most_effective_moves": dict(sorted(self.move_effectiveness.items(),
                                             key=lambda x: x[1],
                                             reverse=True)[:5]),
            "exploration_factor": self.base_exploration_factor + self.exploration_adjustment,
            "confidence_trend": self.confidence_history[-3:] if len(self.confidence_history) >= 3 else self.confidence_history,
            "strategy_confidence": sum(self.move_effectiveness.values()) / len(self.move_effectiveness)
        }