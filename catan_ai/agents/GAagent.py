import numpy as np
import random
from collections.abc import Iterable

from catanatron.models.enums import ActionType, ActionPrompt
from catanatron.models.actions import Action, maritime_trade_possibilities
from catanatron.models.player import Player
from catanatron import Color
from catanatron.game import Game

class GAConstructionAgent(Player):
    def __init__(self, color=Color.BLUE, genome=None):
        super().__init__(color)
        self.color = color
        self.genome = genome if genome is not None else self.random_genome()

    def random_genome(self):
        return np.random.uniform(-1, 1, size=5)

    def decide(self, game: Game, playable_actions: Iterable[Action]):
        if game.state.current_prompt in [ActionPrompt.MOVE_ROBBER, ActionPrompt.DISCARD]:
            return self.handle_special_prompt(game, playable_actions)

        playable_actions = set(playable_actions)
        playable_actions -= set(maritime_trade_possibilities(game.state, self.color))
        playable_actions -= {Action(self.color, ActionType.BUY_DEVELOPMENT_CARD, None)}
        playable_actions = list(playable_actions)

        construction_actions = [
            a for a in playable_actions
            if a.action_type in {ActionType.BUILD_ROAD, ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY}
        ]

        if construction_actions:
            scored = [(a, self.evaluate_action(game, a)) for a in construction_actions]
            return max(scored, key=lambda x: x[1])[0]

        for action in playable_actions:
            if action.action_type == ActionType.END_TURN:
                return action

        if playable_actions:
            return playable_actions[0]

        return Action(self.color, ActionType.END_TURN, None)

    def handle_special_prompt(self, game, playable_actions):
        return random.choice(playable_actions) if playable_actions else None

    def evaluate_action(self, game: Game, action: Action):
        board = game.state.board

        try:
            if action.action_type in [ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY]:
                num_adj = len(board.hexes_adjacent_to_vertex(action.value))
            elif action.action_type == ActionType.BUILD_ROAD:
                num_adj = len(board.get_connected_settlements(action.value))
            else:
                num_adj = 0
        except:
            num_adj = 0

        dice_score = getattr(board, "dice_probability_score", lambda x: random.random())(action.value)

        blocking = self.calculate_blocking(board, action)

        vp = 2 if action.action_type == ActionType.BUILD_CITY else 1 if action.action_type == ActionType.BUILD_SETTLEMENT else 0

        noise = random.random()

        features = [num_adj, dice_score, blocking, vp, noise]
        return np.dot(self.genome, features)

    def calculate_blocking(self, board, action):
        try:
            neighbors = board.get_adjacent_vertices(action.value)
            for v in neighbors:
                owner = board.vertex_owner(v)
                if owner and owner != self.color:
                    return 1.0
        except:
            pass
        return 0.0

class GeneticTrainer:
    def __init__(self, population_size=50, generations=100, mutation_rate=0.1, elite_frac=0.2):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_frac = elite_frac

    def run(self):
        population = [GAConstructionAgent(Color.BLUE) for _ in range(self.population_size)]

        for gen in range(self.generations):
            print(f"\n Generation {gen + 1}")
            scores = []

            for agent in population:
                opponent = GAConstructionAgent(Color.RED)  
                game = Game([agent, opponent])
                game.play()
                reward = self.calculate_fitness(agent, game)
                scores.append((agent, reward))

            scores.sort(key=lambda x: x[1], reverse=True)
            elites = [agent for agent, _ in scores[:int(self.elite_frac * self.population_size)]]

            print(f" Best Fitness: {scores[0][1]:.2f} | Genome: {np.round(scores[0][0].genome, 2)}")

            new_population = elites[:]
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(elites, 2)
                child_genome = self.crossover(parent1.genome, parent2.genome)
                child_genome = self.mutate(child_genome)
                child = GAConstructionAgent(Color.BLUE, genome=child_genome)
                new_population.append(child)

            population = new_population

        return scores[0][0] 

    def calculate_fitness(self, agent, game):
        state = game.state
        key = f"{agent.color}_ACTUAL_VICTORY_POINTS"
        vps = state.player_state.get(key, 0)

        # Bonus for winning
        if game.winning_color() == agent.color:
            return 100 + vps

        num_builds = sum([
            state.player_state.get(f"{agent.color}_NUM_BUILT_ROAD", 0),
            state.player_state.get(f"{agent.color}_NUM_BUILT_SETTLEMENT", 0),
            state.player_state.get(f"{agent.color}_NUM_BUILT_CITY", 0)
        ])

        return vps + 0.5 * num_builds
    def crossover(self, g1, g2):
        alpha = random.random()
        return alpha * g1 + (1 - alpha) * g2

    def mutate(self, genome):
        for i in range(len(genome)):
            if random.random() < self.mutation_rate:
                genome[i] += np.random.normal(0, 0.1)
        return genome

if __name__ == "__main__":
    trainer = GeneticTrainer()
    best_agent = trainer.run()

    print("\n Best Genome Evolved:")
    print(np.round(best_agent.genome, 4))
