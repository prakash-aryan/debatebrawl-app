import random
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum

class Strategy:
    """A debate strategy with rhetorical elements and specific tactics"""
    def __init__(self, genes: Dict[str, float] = None, tactics: List[str] = None):
        # Initialize with default balanced strategy if none provided
        self.genes = genes or {
            "ethos": 0.33,
            "pathos": 0.33,
            "logos": 0.34
        }
        
        self.tactics = tactics or []
        self.fitness = 0.0
        self.effectiveness_history = {}  # Track effectiveness against different move types
        self.normalize_genes()
    
    def normalize_genes(self):
        """Ensure rhetorical elements sum to 1.0"""
        total = sum(self.genes.values())
        if total > 0:
            self.genes = {k: v/total for k, v in self.genes.items()}
    
    def get_dominant_tactics(self, max_tactics: int = 3) -> List[str]:
        """Get the most important tactics in this strategy"""
        return self.tactics[:max_tactics] if self.tactics else []
    
    def record_effectiveness(self, opponent_move: str, score: float):
        """Record how effective this strategy was against a specific opponent move"""
        if opponent_move not in self.effectiveness_history:
            self.effectiveness_history[opponent_move] = []
        self.effectiveness_history[opponent_move].append(score)
    
    def get_average_effectiveness(self, opponent_move: str) -> float:
        """Get average effectiveness against a specific opponent move"""
        if opponent_move in self.effectiveness_history and self.effectiveness_history[opponent_move]:
            return sum(self.effectiveness_history[opponent_move]) / len(self.effectiveness_history[opponent_move])
        return 0.5  # Default neutral effectiveness
    
    def __str__(self):
        """String representation showing elements and tactics"""
        elements = ", ".join([f"{e}={self.genes[e]:.2f}" for e in self.genes])
        tactics = f", tactics: {self.tactics}" if self.tactics else ""
        return f"Strategy({elements}{tactics})"

class DebateStrategyEvolver:
    """Improved Genetic Algorithm for evolving debate strategies with user adaptation"""
    
    def __init__(self, population_size: int):
        self.population_size = population_size
        self.mutation_rate = 0.2
        self.crossover_rate = 0.7
        self.elite_count = max(1, int(population_size * 0.1))  # Keep top 10% as elites
        
        # Counter-strategy mapping - which elements counter others
        self.counter_elements = {
            "pathos": "logos",      # Counter emotional appeals with logic
            "logos": "pathos",      # Counter cold logic with emotional appeals
            "ethos": "logos"        # Counter authority appeals with logical analysis
        }
        
        # Available debate tactics
        self.available_tactics = [
            "present_evidence", "appeal_to_emotion", "logical_reasoning",
            "address_counterarguments", "provide_examples", "use_analogy",
            "cite_expert_opinion", "highlight_consequences", "reframe_issue",
            "ask_rhetorical_questions", "tell_story", "establish_credibility",
            "concede_points", "build_common_ground", "use_statistics"
        ]
        
        # Counter-tactics mapping - which tactics counter others
        self.counter_tactics = {
            "present_evidence": ["address_counterarguments", "reframe_issue"],
            "appeal_to_emotion": ["present_evidence", "logical_reasoning"],
            "logical_reasoning": ["appeal_to_emotion", "ask_rhetorical_questions"],
            "address_counterarguments": ["present_evidence", "provide_examples"],
            "use_analogy": ["logical_reasoning", "present_evidence"],
            "cite_expert_opinion": ["cite_expert_opinion", "question_authority"],
            "highlight_consequences": ["reframe_issue", "address_counterarguments"],
            "ask_rhetorical_questions": ["present_evidence", "logical_reasoning"]
        }
        
        # Strategy effectiveness memory
        self.strategy_memory = {}  # Maps user moves to effective counter-strategies
        
        # Initialize population with diverse strategies
        self.population = self._create_diverse_population()
        
        # Evolution tracking
        self.generation = 0
        self.last_detected_move = None
    
    def _create_diverse_population(self) -> List[Strategy]:
        """Create a diverse initial population of strategies"""
        population = []
        
        # Add strategies with emphasis on each rhetorical element
        for element in ["ethos", "pathos", "logos"]:
            genes = {"ethos": 0.1, "pathos": 0.1, "logos": 0.1}
            genes[element] = 0.8  # Make this element dominant
            
            # Choose 3-5 random tactics
            num_tactics = random.randint(3, 5)
            selected_tactics = random.sample(self.available_tactics, num_tactics)
            
            population.append(Strategy(genes, selected_tactics))
        
        # Add counter-strategy focused individuals
        for counter_from, counter_to in self.counter_elements.items():
            genes = {"ethos": 0.1, "pathos": 0.1, "logos": 0.1}
            genes[counter_to] = 0.7  # Emphasize the counter element
            
            # Add effective counter tactics
            counter_tacs = []
            for tactic, counters in self.counter_tactics.items():
                if any(t in counters for t in self.available_tactics):
                    counter_tacs.extend(counters)
            
            selected_tactics = random.sample(counter_tacs if counter_tacs else self.available_tactics, 
                                           min(3, len(counter_tacs) if counter_tacs else len(self.available_tactics)))
            
            population.append(Strategy(genes, selected_tactics))
        
        # Fill the rest with random strategies
        while len(population) < self.population_size:
            population.append(self.create_random_strategy())
        
        return population
    
    def create_random_strategy(self) -> Strategy:
        """Create a random debate strategy"""
        # Generate random weights for rhetorical elements
        genes = {
            "ethos": random.random(),
            "pathos": random.random(),
            "logos": random.random()
        }
        
        # Select random debate tactics (3-7)
        num_tactics = random.randint(3, min(7, len(self.available_tactics)))
        selected_tactics = random.sample(self.available_tactics, num_tactics)
        
        return Strategy(genes, selected_tactics)
    
    def evolve(self, fitness_scores: List[float], detected_move: str = None) -> Strategy:
        """
        Evolve the population based on fitness scores and adapt to user's detected move
        
        Args:
            fitness_scores: List of fitness values for each strategy
            detected_move: User's detected debate move to adapt against
        """
        if len(fitness_scores) != self.population_size:
            raise ValueError(f"Fitness scores count ({len(fitness_scores)}) must match population size ({self.population_size})")
        
        # Update last detected move
        self.last_detected_move = detected_move
        
        # Update fitness values and record effectiveness against detected move
        for i, strategy in enumerate(self.population):
            strategy.fitness = fitness_scores[i]
            if detected_move:
                strategy.record_effectiveness(detected_move, fitness_scores[i])
        
        # Sort population by fitness (descending)
        self.population.sort(key=lambda s: s.fitness, reverse=True)
        
        # Save best strategy for this move type in memory
        if detected_move and self.population[0].fitness > 0.6:
            if detected_move not in self.strategy_memory:
                self.strategy_memory[detected_move] = []
            # Only keep limited memory (last 3 effective strategies)
            self.strategy_memory[detected_move] = (self.strategy_memory[detected_move] + 
                                                 [self._deep_copy_strategy(self.population[0])])[-3:]
        
        # Create new population starting with elites
        new_population = [self._deep_copy_strategy(strategy) for strategy in self.population[:self.elite_count]]
        
        # If we have memory of effective strategies against this move, include them
        if detected_move and detected_move in self.strategy_memory:
            for stored_strategy in self.strategy_memory[detected_move]:
                if len(new_population) < self.population_size:
                    # Apply some mutation to avoid getting stuck
                    mutated_memory = self._mutate(self._deep_copy_strategy(stored_strategy))
                    new_population.append(mutated_memory)
        
        # Fill the rest through selection, crossover and mutation
        while len(new_population) < self.population_size:
            # Selection - higher tournament size (4 instead of 3) for more pressure
            parent1 = self._tournament_selection(4, detected_move)
            parent2 = self._tournament_selection(4, detected_move)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = self._deep_copy_strategy(parent1)
            
            # Targeted mutation based on detected move
            child = self._mutate(child, detected_move)
            
            # Add to new population
            new_population.append(child)
        
        # Replace old population
        self.population = new_population
        self.generation += 1
        
        # Return the best strategy
        return self.population[0]
    
    def _tournament_selection(self, tournament_size: int, detected_move: str = None) -> Strategy:
        """
        Select a parent using tournament selection, favoring strategies that counter detected move
        """
        # Select random strategies for tournament
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        
        if detected_move:
            # Adjust fitness based on effectiveness against the detected move
            adjusted_tournament = []
            for strategy in tournament:
                effectiveness = strategy.get_average_effectiveness(detected_move)
                # Create a copy with adjusted fitness
                adjusted = self._deep_copy_strategy(strategy)
                # Boost fitness for strategies effective against this move
                adjusted.fitness = strategy.fitness * (1.0 + effectiveness * 0.5)
                adjusted_tournament.append(adjusted)
            tournament = adjusted_tournament
        
        # Return the fittest
        return max(tournament, key=lambda s: s.fitness)
    
    def _crossover(self, parent1: Strategy, parent2: Strategy) -> Strategy:
        """Perform crossover between two parent strategies"""
        # Two-point crossover for genes
        child_genes = {}
        gene_keys = list(parent1.genes.keys())
        
        if len(gene_keys) >= 3:
            # Two-point crossover
            points = sorted(random.sample(range(1, len(gene_keys)), 2))
            
            for i, gene in enumerate(gene_keys):
                if i < points[0] or i >= points[1]:
                    child_genes[gene] = parent1.genes[gene]
                else:
                    child_genes[gene] = parent2.genes[gene]
        else:
            # Single-point crossover for fewer genes
            for gene in gene_keys:
                child_genes[gene] = parent1.genes[gene] if random.random() < 0.5 else parent2.genes[gene]
        
        # Tactics crossover - select from both parents
        all_tactics = list(set(parent1.tactics + parent2.tactics))
        
        if all_tactics:
            # Take some tactics from each parent with preference for parent1 if it has higher fitness
            num_tactics = min(len(all_tactics), random.randint(3, 7))
            weights = [2 if tactic in parent1.tactics else 1 for tactic in all_tactics]
            if parent2.fitness > parent1.fitness:
                weights = [2 if tactic in parent2.tactics else 1 for tactic in all_tactics]
                
            child_tactics = random.choices(all_tactics, weights=weights, k=num_tactics)
            # Remove duplicates while preserving order
            child_tactics = list(dict.fromkeys(child_tactics))
        else:
            child_tactics = []
        
        return Strategy(child_genes, child_tactics)
    
    def _mutate(self, strategy: Strategy, detected_move: str = None) -> Strategy:
        """Mutate a strategy, targeting improvements against detected user move"""
        # Clone the strategy
        mutated = self._deep_copy_strategy(strategy)
        
        # If we have a detected move, bias mutation toward counter-elements
        if detected_move:
            # Determine dominant element in user's approach
            move_element = self._get_move_dominant_element(detected_move)
            if move_element and move_element in self.counter_elements:
                # Get the counter element
                counter_element = self.counter_elements[move_element]
                
                # Higher chance to increase the counter element
                if random.random() < 0.6:  # 60% chance
                    # Boost the counter element
                    current = mutated.genes.get(counter_element, 0.33)
                    mutated.genes[counter_element] = min(0.9, current + random.uniform(0.1, 0.3))
                    
                    # Also more likely to add counter tactics
                    counter_tacs = self._get_counter_tactics(detected_move)
                    if counter_tacs and random.random() < 0.7:
                        for tactic in counter_tacs:
                            if tactic not in mutated.tactics and random.random() < 0.5:
                                mutated.tactics.append(tactic)
        
        # Standard mutation for rhetorical elements
        for gene in mutated.genes:
            if random.random() < self.mutation_rate:
                # Apply significant adjustment (+/- 30%)
                adjustment = random.uniform(-0.3, 0.3)
                mutated.genes[gene] = max(0.1, min(0.9, mutated.genes[gene] + adjustment))
        
        # Normalize genes
        mutated.normalize_genes()
        
        # Standard mutation for tactics
        
        # Possibly add a new tactic
        if random.random() < self.mutation_rate and len(mutated.tactics) < 7:
            unused_tactics = [t for t in self.available_tactics if t not in mutated.tactics]
            if unused_tactics:
                mutated.tactics.append(random.choice(unused_tactics))
        
        # Possibly remove a tactic
        if random.random() < self.mutation_rate and len(mutated.tactics) > 3:
            # Remove a random tactic
            tactic_to_remove = random.choice(mutated.tactics)
            mutated.tactics.remove(tactic_to_remove)
        
        # Possibly reorder tactics
        if random.random() < self.mutation_rate and len(mutated.tactics) > 1:
            random.shuffle(mutated.tactics)
        
        return mutated
    
    def _deep_copy_strategy(self, strategy: Strategy) -> Strategy:
        """Create a deep copy of a strategy"""
        new_strategy = Strategy(
            genes={k: v for k, v in strategy.genes.items()},
            tactics=strategy.tactics.copy()
        )
        new_strategy.fitness = strategy.fitness
        new_strategy.effectiveness_history = {
            k: v.copy() for k, v in strategy.effectiveness_history.items()
        }
        return new_strategy
    
    def _get_move_dominant_element(self, move_type: str) -> str:
        """Determine which rhetorical element dominates a move type"""
        # Map moves to likely dominant elements
        move_elements = {
            "present_evidence": "logos",
            "appeal_to_emotion": "pathos",
            "use_logical_reasoning": "logos",
            "establish_credibility": "ethos",
            "cite_expert_opinion": "ethos",
            "highlight_consequences": "pathos",
            "tell_story": "pathos",
            "ask_rhetorical_questions": "pathos"
        }
        return move_elements.get(move_type, "logos")  # Default to logos
    
    def _get_counter_tactics(self, move_type: str) -> List[str]:
        """Get tactics that counter a specific move type"""
        return self.counter_tactics.get(move_type, [])
    
    def get_best_strategy(self, fitness_scores: List[float], detected_move: str = None) -> Strategy:
        """Get the best strategy, considering effectiveness against detected move"""
        if not fitness_scores or len(fitness_scores) != len(self.population):
            # Return first strategy if no valid fitness scores
            return self.population[0]
        
        # Update fitness values
        for i, strategy in enumerate(self.population):
            strategy.fitness = fitness_scores[i]
        
        if detected_move:
            # Consider both fitness and effectiveness against this move type
            best_strategy = None
            best_score = -1
            
            for strategy in self.population:
                effectiveness = strategy.get_average_effectiveness(detected_move)
                # Combined score weights both general fitness and specific counter-effectiveness
                combined_score = strategy.fitness * 0.7 + effectiveness * 0.3
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_strategy = strategy
            
            return best_strategy
        else:
            # Just return the strategy with highest fitness
            return max(self.population, key=lambda s: s.fitness)