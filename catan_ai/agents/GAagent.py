import random
from collections.abc import Iterable
from typing import Any, Dict, List, Tuple

import numpy as np
from catanatron.game import Game
from catanatron.models.actions import Action, maritime_trade_possibilities
from catanatron.models.enums import ActionPrompt, ActionType
from catanatron.models.player import Color, Player


class GeneticAlgorithmAgent(Player):
    def __init__(
        self,
        color=Color.BLUE,
        population_size=30,
        generations=20,
        mutation_rate=0.15,
        crossover_rate=0.8,
        tournament_size=3,
        weights=None,
        evaluation_games=3,
    ):
        super().__init__(color)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.evaluation_games = evaluation_games
        self.feature_count = 14
        if weights is None:
            self.weights = np.random.uniform(-1, 1, self.feature_count)
        else:
            self.weights = np.array(weights)
        self.population = self._initialize_population()
        self.best_individual = self.weights
        self.best_fitness = float("-inf")
        self.fitness_history = []
        self.position_quality_cache = {}

    def decide(self, game: Game, playable_actions: Iterable[Action]) -> Action:
        playable_actions = set(playable_actions)
        playable_actions -= set(maritime_trade_possibilities(game.state, self.color))
        playable_actions -= {Action(self.color, ActionType.BUY_DEVELOPMENT_CARD, None)}
        playable_actions = list(playable_actions)
        if (
            game.state.current_prompt == ActionPrompt.MOVE_ROBBER
            or game.state.current_prompt == ActionPrompt.DISCARD
        ):
            return self.handle_special_prompt(game, playable_actions)
        if not playable_actions:
            return None
        action_scores = []
        for action in playable_actions:
            features = self._extract_features(game, action)
            score = np.dot(self.best_individual, features)
            action_scores.append((action, score))
        best_action = max(action_scores, key=lambda x: x[1])[0]
        return best_action

    def handle_special_prompt(self, game, playable_actions):
        if game.state.current_prompt == ActionPrompt.MOVE_ROBBER:
            best_action = None
            max_vp = -1
            for action in playable_actions:
                if action.action_type == ActionType.MOVE_ROBBER:
                    if action.value[1] is not None:
                        target_color = action.value[1]
                        key = f"{target_color}_ACTUAL_VICTORY_POINTS"
                        vp = game.state.player_state.get(key, 0)
                        if vp > max_vp:
                            max_vp = vp
                            best_action = action
            if best_action:
                return best_action
        return random.choice(playable_actions) if playable_actions else None

    def _initialize_population(self) -> List[np.ndarray]:
        return [np.random.uniform(-1, 1, self.feature_count) for _ in range(self.population_size)]

    def _extract_features(self, game: Game, action: Action) -> np.ndarray:
        features = np.zeros(self.feature_count)
        action_type_str = str(action.action_type)
        action_type_map = {
            "ActionType.BUILD_ROAD": 0,
            "ActionType.BUILD_SETTLEMENT": 1,
            "ActionType.BUILD_CITY": 2,
            "ActionType.END_TURN": 3,
        }
        action_type_idx = action_type_map.get(action_type_str, 0)
        features[action_type_idx] = 1
        key = f"{self.color}_ACTUAL_VICTORY_POINTS"
        features[4] = game.state.player_state.get(key, 0) / 10.0
        resource_counts = self._get_resource_counts(game)
        features[5] = sum(1 for count in resource_counts if count > 0) / 5.0
        features[6] = sum(resource_counts) / 20.0
        features[7] = self._count_buildings(game, ActionType.BUILD_SETTLEMENT) / 5.0
        features[8] = self._count_buildings(game, ActionType.BUILD_CITY) / 4.0
        if (
            action.action_type == ActionType.BUILD_SETTLEMENT
            or action.action_type == ActionType.BUILD_CITY
        ):
            position = action.value
            features[9] = self._evaluate_position_quality(game, position) / 10.0
            features[10] = self._evaluate_resource_scarcity(game, position) / 10.0
        if (
            action.action_type == ActionType.BUILD_ROAD
            or action.action_type == ActionType.BUILD_SETTLEMENT
        ):
            features[11] = self._evaluate_blocking_potential(game, action) / 5.0
        features[12] = min(game.state.num_turns / 50.0, 1.0)
        if action.action_type == ActionType.BUILD_ROAD:
            features[13] = self._evaluate_longest_road_potential(game, action) / 5.0
        return features

    def _get_resource_counts(self, game: Game) -> List[int]:
        key = f"{self.color}_RESOURCES"
        return game.state.player_state.get(key, [0, 0, 0, 0, 0])

    def _count_buildings(self, game: Game, building_type: ActionType) -> int:
        if building_type == ActionType.BUILD_SETTLEMENT:
            key = f"{self.color}_SETTLEMENTS"
        elif building_type == ActionType.BUILD_CITY:
            key = f"{self.color}_CITIES"
        else:
            return 0
        return len(game.state.player_state.get(key, []))

    def _evaluate_position_quality(self, game: Game, position) -> float:
        cache_key = str(position)
        if cache_key in self.position_quality_cache:
            return self.position_quality_cache[cache_key]
        quality = random.uniform(3.0, 8.0)
        self.position_quality_cache[cache_key] = quality
        return quality

    def _evaluate_resource_scarcity(self, game: Game, position) -> float:
        return random.uniform(3.0, 8.0)

    def _evaluate_blocking_potential(self, game: Game, action: Action) -> float:
        return random.uniform(1.0, 4.0)

    def _evaluate_longest_road_potential(self, game: Game, action: Action) -> float:
        return random.uniform(1.0, 5.0)

    def _evaluate_fitness(self, individual: np.ndarray) -> float:
        original_best = self.best_individual.copy()
        self.best_individual = individual
        wins = 0
        total_vps = 0
        for _ in range(self.evaluation_games):
            try:
                opponent_color = Color.RED if self.color != Color.RED else Color.BLUE
                self_opponent = GeneticAlgorithmAgent(color=opponent_color)
                self_opponent.best_individual = individual.copy()
                game = Game([self, self_opponent])
                winner = game.play()
                if winner == self.color:
                    wins += 1
                key = f"{self.color}_ACTUAL_VICTORY_POINTS"
                vps = game.state.player_state.get(key, 0)
                total_vps += vps
            except Exception as e:
                print(f"Error during fitness evaluation: {e}")
                self.best_individual = original_best
                return 1.0
        self.best_individual = original_best
        avg_vps = total_vps / self.evaluation_games
        win_rate = wins / self.evaluation_games
        fitness = win_rate * 10.0 + avg_vps / 10.0
        return fitness

    def _selection(
        self, population: List[np.ndarray], fitness_scores: List[float]
    ) -> List[np.ndarray]:
        selected = []
        for _ in range(len(population)):
            tournament_indices = random.sample(
                range(len(population)), min(self.tournament_size, len(population))
            )
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
        return selected

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.crossover_rate:
            crossover_points = sorted(random.sample(range(1, len(parent1)), 2))
            child1 = np.copy(parent1)
            child2 = np.copy(parent2)
            child1[crossover_points[0] : crossover_points[1]] = parent2[
                crossover_points[0] : crossover_points[1]
            ]
            child2[crossover_points[0] : crossover_points[1]] = parent1[
                crossover_points[0] : crossover_points[1]
            ]
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()

    def _mutation(self, individual: np.ndarray) -> np.ndarray:
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutation_scale = (
                    0.5 * (1.0 - self.best_fitness / 15.0) if self.best_fitness > 0 else 0.5
                )
                mutated[i] += random.uniform(-mutation_scale, mutation_scale)
                mutated[i] = np.clip(mutated[i], -1, 1)
        return mutated

    def train(self, num_generations: int = None) -> Dict[str, Any]:
        if num_generations is None:
            num_generations = self.generations
        all_generation_weights = {}
        print(
            f"Starting training with population size {self.population_size}, "
            f"{num_generations} generations, mutation rate {self.mutation_rate}, "
            f"crossover rate {self.crossover_rate}, tournament size {self.tournament_size}"
        )
        print(f"Each individual will be evaluated by playing {self.evaluation_games} games")
        print("Initializing population with simplified fitness evaluation...")
        for i, ind in enumerate(self.population):
            action_type_map = {
                "ActionType.BUILD_ROAD": 0,
                "ActionType.BUILD_SETTLEMENT": 1,
                "ActionType.BUILD_CITY": 2,
                "ActionType.END_TURN": 3,
            }
            road_idx = action_type_map["ActionType.BUILD_ROAD"]
            settlement_idx = action_type_map["ActionType.BUILD_SETTLEMENT"]
            city_idx = action_type_map["ActionType.BUILD_CITY"]
            settlement_weight = ind[settlement_idx]
            city_weight = ind[city_idx]
            road_weight = ind[road_idx]
            position_quality_weight = ind[9]
            fitness = (
                settlement_weight
                + city_weight
                + 0.5 * road_weight
                + position_quality_weight
                - np.std(ind)
            )
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_individual = ind.copy()
            if (i + 1) % 10 == 0:
                print(f"Initialized {i + 1}/{self.population_size} individuals")
        print(f"Initial best fitness: {self.best_fitness:.4f}")
        print("Starting actual training with game-based fitness evaluation...")
        for generation in range(num_generations):
            print(f"Generation {generation + 1}/{num_generations}...")
            fitness_scores = []
            for i, ind in enumerate(self.population):
                fitness = self._evaluate_fitness(ind)
                fitness_scores.append(fitness)
                print(f"  Individual {i + 1}/{self.population_size}: Fitness = {fitness:.4f}")
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[max_fitness_idx]
                self.best_individual = self.population[max_fitness_idx].copy()
            avg_fitness = np.mean(fitness_scores)
            self.fitness_history.append(
                {
                    "generation": generation,
                    "max_fitness": max(fitness_scores),
                    "avg_fitness": avg_fitness,
                    "min_fitness": min(fitness_scores),
                }
            )
            print(
                f"  Generation {generation + 1}/{num_generations}: "
                f"Best Fitness = {self.best_fitness:.4f}, "
                f"Avg Fitness = {avg_fitness:.4f}"
            )
            all_generation_weights[f"generation_{generation + 1}"] = self.best_individual.copy()
            if self.best_fitness >= 15.0:
                print(f"Reached excellent fitness of {self.best_fitness:.4f}. Stopping early.")
                break
            selected = self._selection(self.population, fitness_scores)
            new_population = []
            new_population.append(self.best_individual.copy())
            while len(new_population) < self.population_size:
                parent1 = random.choice(selected)
                parent2 = random.choice(selected)
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutation(child1)
                child2 = self._mutation(child2)
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            self.population = new_population[: self.population_size]
        np.save("all_generation_weights.npy", all_generation_weights)
        print("All generation weights saved to 'all_generation_weights.npy'")
        self.save_weights("final_weights.npy")
        print(f"Training completed. Final best fitness: {self.best_fitness:.4f}")
        print("Final weights saved to 'final_weights.npy'")
        return {
            "best_individual": self.best_individual,
            "best_fitness": self.best_fitness,
            "fitness_history": self.fitness_history,
            "all_weights": all_generation_weights,
        }

    def load_generation_weights(self, filename: str, generation: str) -> None:
        all_weights = np.load(filename, allow_pickle=True).item()
        if generation in all_weights:
            self.best_individual = all_weights[generation]
            print(f"Loaded weights from {generation}")
        else:
            raise ValueError(f"Generation {generation} not found in weights file")

    def save_weights(self, filename: str) -> None:
        np.save(filename, self.best_individual)
        print(f"Weights saved to {filename}")

    def load_weights(self, filename: str) -> None:
        self.best_individual = np.load(filename, allow_pickle=True)
        print(f"Weights loaded from {filename}")


if __name__ == "__main__":
    agent = GeneticAlgorithmAgent(
        color=Color.WHITE,
        population_size=20,
        generations=50,
        mutation_rate=0.15,
        crossover_rate=0.8,
        tournament_size=3,
        evaluation_games=5,
    )
    print("Training enhanced genetic algorithm agent...")
    results = agent.train()
    print(f"Best fitness achieved: {results['best_fitness']}")
    print("Training history:")
    for entry in agent.fitness_history:
        print(
            f"Generation {entry['generation'] + 1}: "
            f"Max={entry['max_fitness']:.4f}, "
            f"Avg={entry['avg_fitness']:.4f}, "
            f"Min={entry['min_fitness']:.4f}"
        )

