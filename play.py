import sys

from catanatron import Color, Game

from catan_ai.agents.GAagent import GeneticAlgorithmAgent
from catan_ai.agents.random_agent import RandomAgent


def main() -> None:
    num_games = int(sys.argv[1]) if len(sys.argv) >= 2 else 100

    def create_ga_agent(color):
        agent = GeneticAlgorithmAgent(color)
        try:
            agent.load_weights("final_weights.npy")
            print("Loaded pre-trained weights for genetic algorithm agent")
        except FileNotFoundError:
            print("No pre-trained weights found. Using untrained agent.")
        return agent

    players = [RandomAgent(Color.BLUE), create_ga_agent(Color.WHITE)]

    scorecard = {}

    for _ in range(num_games):
        game = Game(players)
        game.TURNS_LIMIT = 5000000
        winner = game.play()

        scorecard[winner] = scorecard.get(winner, 0) + 1

    print("Player\tWins")
    print("------\t----")
    for player, wins in scorecard.items():
        name = player.name if player is not None else "None"
        print(f"{name}\t{wins}")


if __name__ == "__main__":
    main()
