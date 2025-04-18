from catanatron import Color, Game

from catan_ai.agents import DQNTrainAgent, RandomAgent

MODEL_PATH = "dqn_model.pt"
NUM_GAMES = 20_000


def main() -> None:
    dqn_agent = DQNTrainAgent(Color.BLUE)

    players = [
        dqn_agent,
        RandomAgent(Color.RED),
    ]

    scorecard = {}

    for idx in range(NUM_GAMES):
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

        train_str = f"Game: {idx / NUM_GAMES:.0%}, "
        train_str += f"Win vs. Red: {player_score:.0%}, "
        train_str += f"Win vs. All: {all_score:.0%}, "
        train_str += f"Decay: {dqn_agent.eps_threshold}"
        print(train_str)

    dqn_agent.save(MODEL_PATH)


if __name__ == "__main__":
    main()
