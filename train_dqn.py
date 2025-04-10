import os
import pickle

from catanatron import Color, Game, game
from catanatron.models.player import RandomPlayer
from catanatron_server.utils import open_link

from catan_ai.agents import DQNTrainAgent, RandomAgent

game.TURNS_LIMIT = 1000

MODEL_PATH = "model.py"
BOARD_PATH = "board.pickle"


def main() -> None:
    dqn_agent = DQNTrainAgent(Color.BLUE, path=MODEL_PATH)

    players = [
        dqn_agent,
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

    for idx in range(1000):
        game = Game(players, catan_map=board)
        winner = game.play()

        dqn_agent.game_over(game)
        scorecard[winner] = scorecard.get(winner, 0) + 1

        try:
            player_score = scorecard.get(Color.BLUE, 0) / sum(
                score for color, score in scorecard.items() if color is not None
            )
        except Exception:
            player_score = 0
        all_score = scorecard.get(Color.BLUE, 0) / sum(scorecard.values())
        print(
            f"Game: {idx / 1000:.0%}, Win vs. Red: {player_score:.0%}, Win vs. All: {all_score:.0%}"
        )

    dqn_agent.save(MODEL_PATH)

    print("Player\tWins")
    print("------\t----")
    for player, wins in scorecard.items():
        name = player.name if player is not None else "None"
        print(f"{name}\t{wins}")

    game.state.players = [RandomPlayer(player.color) for player in game.state.players]
    open_link(game)


if __name__ == "__main__":
    main()
