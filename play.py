import os
import pickle
import sys

from catanatron import Color, Game, game
from catanatron.models.player import RandomPlayer
from catanatron_server.utils import open_link

from catan_ai.agents import DQNAgent, RandomAgent

game.TURNS_LIMIT = 1000

MODEL_PATH = "model.pt"
BOARD_PATH = "board.pickle"


def main() -> None:
    num_games = int(sys.argv[1]) if len(sys.argv) >= 2 else 1

    players = [
        DQNAgent(Color.BLUE, path=MODEL_PATH),
        RandomAgent(Color.RED),
    ]

    if os.path.exists(BOARD_PATH):
        with open(BOARD_PATH, "rb") as file:
            board = pickle.load(file)
    else:
        board = Game(players).state.board.map
        with open(BOARD_PATH, "wb") as file:
            pickle.dump(board, file)

    scorecard = {}

    for _ in range(num_games):
        game = Game(players, catan_map=board)
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
