import random
from collections.abc import Iterable

from catanatron.game import Game
from catanatron.models.actions import Action, maritime_trade_possibilities
from catanatron.models.enums import ActionType
from catanatron.models.player import Player


class RandomAgent(Player):
    def decide(self, game: Game, playable_actions: Iterable[Action]) -> Action:
        playable_actions = set(playable_actions)

        robber_action = Action(
            self.color, ActionType.MOVE_ROBBER, game.state.board.robber_coordinate
        )
        if robber_action in playable_actions:
            return robber_action

        playable_actions -= set(maritime_trade_possibilities(game.state, self.color))
        playable_actions -= {Action(self.color, ActionType.BUY_DEVELOPMENT_CARD, None)}

        playable_actions = list(playable_actions)

        action = random.choice(playable_actions)
        return action
