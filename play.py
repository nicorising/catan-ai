import sys

from catanatron import Color, Game, RandomPlayer
from catanatron_server.utils import open_link

from catan_ai.agents import DQNAgent, GeneticAlgorithmAgent, RandomAgent

MODEL_PATH = "model.pt"


def create_ga_agent(color):
    agent = GeneticAlgorithmAgent(color)
    try:
        agent.load_weights("final_weights.npy")
        print("Loaded pre-trained weights for genetic algorithm agent")
    except FileNotFoundError:
        print("No pre-trained weights found. Using untrained agent.")
    return agent


def main() -> None:
    num_games = int(sys.argv[1]) if len(sys.argv) >= 2 else 100

    players = [
        DQNAgent(Color.BLUE, path=MODEL_PATH),
        RandomAgent(Color.RED),
    ]

    scorecard = {}

    for _ in range(num_games):
        game = Game(players)
        winner = game.play()

        scorecard[winner] = scorecard.get(winner, 0) + 1

    print("Player\tWins")
    print("------\t----")
    for player, wins in scorecard.items():
        name = player.name if player is not None else "None"
        print(f"{name}\t{wins}")

    game.state.players = [RandomPlayer(player.color) for player in game.state.players]
    open_link(game)


if __name__ == "__main__":
    main()
