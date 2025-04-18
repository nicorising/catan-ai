import sys

from catanatron import Color, Game, RandomPlayer
from sqlalchemy.exc import OperationalError

from catan_ai.agents import MCTS, DQNAgent, GeneticAlgorithmAgent, RandomAgent, TDAgent


def main() -> None:
    match sys.argv[1].lower() if len(sys.argv) >= 2 else "":
        case "mcts":
            agent_name = "MCTS"
            agent = MCTS(Color.BLUE)
            print("Loaded MCTS agent")
        case "td":
            agent_name = "TD"
            agent = TDAgent(Color.BLUE)
            agent.load()
            print("Loaded pre-trained TD agent")
        case "ga":
            agent_name = "GA"
            agent = GeneticAlgorithmAgent(Color.BLUE)
            try:
                agent.load_weights("final_weights.npy")
                print("Loaded pre-trained weights for genetic algorithm agent")
            except FileNotFoundError:
                print("No pre-trained weights found. Using untrained agent")
        case "dqn":
            agent_name = "DQN"
            agent = DQNAgent(Color.BLUE, path="dqn_model.pt")
            print("Loaded pre-trained DQN agent")
        case _:
            agent_name = "Random"
            agent = RandomAgent(Color.BLUE)
            print("No model specified/unknown name used, loading random agent")

    num_games = int(sys.argv[2]) if len(sys.argv) >= 3 else 1

    players = [
        agent,
        RandomAgent(Color.RED),
    ]

    scorecard = {}

    for _ in range(num_games):
        game = Game(players)
        winner = game.play()

        scorecard[winner] = scorecard.get(winner, 0) + 1

    print("Player\tWins")
    print("------\t----")
    print(f"{agent_name}\t{scorecard.get(Color.BLUE, 0)}")
    print(f"Random\t{scorecard.get(Color.RED, 0)}")
    print(f"Draw\t{scorecard.get(None, 0)}")

    # Fix to avoid bug uploading custom models to Catanatron server
    game.state.players = [RandomPlayer(player.color) for player in game.state.players]

    try:
        from catanatron_server.utils import open_link

        open_link(game)
    except ModuleNotFoundError | OperationalError:
        print("Unable to visualize game state")


if __name__ == "__main__":
    main()
