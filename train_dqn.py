from catanatron import Color, Game, game
from catanatron.models.player import RandomPlayer
from catanatron_server.utils import open_link

from catan_ai.agents import DQNTrainAgent, RandomAgent

game.TURNS_LIMIT = 1000

MODEL_PATH = "model.pt"


def main() -> None:
    dqn_agent = DQNTrainAgent(Color.BLUE, path=None)
    # dqn_agent = DQNTrainAgent(Color.BLUE, path=MODEL_PATH)

    players = [
        dqn_agent,
        RandomAgent(Color.RED),
    ]

    scorecard = {}

    for idx in range(1_000):
        game = Game(players)
        winner = game.play()

        dqn_agent.game_over(game)
        scorecard[winner] = scorecard.get(winner, 0) + 1

        if idx % 100 == 0:
            dqn_agent.save(MODEL_PATH)

        try:
            player_score = scorecard.get(Color.BLUE, 0) / sum(
                score for color, score in scorecard.items() if color is not None
            )
        except Exception:
            player_score = 0
        all_score = scorecard.get(Color.BLUE, 0) / sum(scorecard.values())
        print(
            f"Game: {idx / 1_000:.0%}, Win vs. Red: {player_score:.0%}, Win vs. All: {all_score:.0%}, Decay: {dqn_agent.eps_threshold}"
        )

    dqn_agent.save(MODEL_PATH)

    print("Player\tWins")
    print("------\t----")
    for player, wins in scorecard.items():
        name = player.name if player is not None else "None"
        print(f"{name}\t{wins}")

    dqn_agent.plot_loss()

    game.state.players = [RandomPlayer(player.color) for player in game.state.players]
    open_link(game)


if __name__ == "__main__":
    main()
