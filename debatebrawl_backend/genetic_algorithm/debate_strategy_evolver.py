import random
from typing import List, Tuple

class Strategy:
    def __init__(self, genes: List[float]):
        self.genes = genes

    def __str__(self):
        return f"Strategy(ethos={self.genes[0]:.2f}, pathos={self.genes[1]:.2f}, logos={self.genes[2]:.2f})"

class DebateStrategyEvolver:
    def __init__(self, population_size: int, gene_length: int):
        self.population_size = population_size
        self.gene_length = gene_length
        self.population = [self.create_random_strategy() for _ in range(population_size)]
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7

    def create_random_strategy(self) -> Strategy:
        genes = [random.random() for _ in range(self.gene_length)]
        total = sum(genes)
        normalized_genes = [gene / total for gene in genes]
        return Strategy(normalized_genes)

    def evolve(self, fitness_scores: List[float]):
        total_fitness = sum(fitness_scores)
        normalized_fitness = [score / total_fitness for score in fitness_scores]

        new_population = []
        for _ in range(self.population_size):
            parent1 = self.select_parent(normalized_fitness)
            parent2 = self.select_parent(normalized_fitness)
            
            if random.random() < self.crossover_rate:
                child = self.crossover(parent1, parent2)
            else:
                child = parent1
            
            child = self.mutate(child)
            new_population.append(child)

        self.population = new_population

    def select_parent(self, normalized_fitness: List[float]) -> Strategy:
        return random.choices(self.population, weights=normalized_fitness, k=1)[0]

    def crossover(self, parent1: Strategy, parent2: Strategy) -> Strategy:
        split = random.randint(1, self.gene_length - 1)
        child_genes = parent1.genes[:split] + parent2.genes[split:]
        total = sum(child_genes)
        normalized_genes = [gene / total for gene in child_genes]
        return Strategy(normalized_genes)

    def mutate(self, strategy: Strategy) -> Strategy:
        new_genes = []
        for gene in strategy.genes:
            if random.random() < self.mutation_rate:
                new_gene = gene + random.uniform(-0.1, 0.1)
                new_genes.append(max(0, min(1, new_gene)))
            else:
                new_genes.append(gene)
        total = sum(new_genes)
        normalized_genes = [gene / total for gene in new_genes]
        return Strategy(normalized_genes)

    def get_best_strategy(self, fitness_scores: List[float]) -> Strategy:
        best_index = fitness_scores.index(max(fitness_scores))
        return self.population[best_index]

    def get_average_fitness(self, fitness_scores: List[float]) -> float:
        return sum(fitness_scores) / len(fitness_scores)

    def get_population_diversity(self) -> float:
        if not self.population:
            return 0.0
        
        avg_genes = [sum(strategy.genes[i] for strategy in self.population) / self.population_size 
                     for i in range(self.gene_length)]
        
        diversity = sum(sum((strategy.genes[i] - avg_genes[i])**2 
                            for i in range(self.gene_length)) 
                        for strategy in self.population) / self.population_size
        
        return diversity

    def apply_strategy(self, strategy: Strategy, argument: str) -> str:
        ethos, pathos, logos = strategy.genes
        
        enhanced_argument = argument

        # Apply ethos (credibility)
        if ethos > 0.6:
            enhanced_argument = f"As an expert in this field, I can confidently say that {enhanced_argument}"
        elif ethos > 0.3:
            enhanced_argument = f"Based on my research and experience, {enhanced_argument}"

        # Apply pathos (emotional appeal)
        if pathos > 0.6:
            enhanced_argument += " This issue deeply affects all of us and our future generations."
        elif pathos > 0.3:
            enhanced_argument += " We should consider the human impact of this issue."

        # Apply logos (logical reasoning)
        if logos > 0.6:
            enhanced_argument += " The data clearly supports this conclusion, as evidenced by multiple peer-reviewed studies."
        elif logos > 0.3:
            enhanced_argument += " This argument is supported by several key facts and logical inferences."

        return enhanced_argument