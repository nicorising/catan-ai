import sys

from catanatron import Color, Game, RandomPlayer

from catan_ai.agents.random_agent import RandomAgent


def main() -> None:
    num_games = int(sys.argv[1]) if len(sys.argv) >= 2 else 1

    players = [
        RandomAgent(Color.BLUE),
        RandomPlayer(Color.RED),
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


if __name__ == "__main__":
    main()
