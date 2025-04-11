import sys
import time

from catanatron import Color, Game

from catan_ai.agents.mcts import MCTS
from catan_ai.agents.random_agent import RandomAgent


def timer(games, start, finish):
    print(f"runtime in seconds for {games} is {finish - start}")


def main() -> None:
    num_games = int(sys.argv[1]) if len(sys.argv) >= 2 else 1

    players = [MCTS(Color.BLUE), RandomAgent(Color.RED)]

    scorecard = {}

    num_games = 10
    start_time = time.time()
    for _ in range(num_games):
        game = Game(players)
        winner = game.play()
        print(f"{winner} won")

        scorecard[winner] = scorecard.get(winner, 0) + 1

    print("Player\tWins")
    print("------\t----")
    for player, wins in scorecard.items():
        name = player.name if player is not None else "None"
        print(f"{name}\t{wins}")
    finish_time = time.time()
    timer(num_games, start_time, finish_time)


if __name__ == "__main__":
    main()
