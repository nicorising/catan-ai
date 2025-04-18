import random
from collections.abc import Iterable

from catanatron import Game
from catanatron.models.actions import Action, maritime_trade_possibilities
from catanatron.models.enums import ActionPrompt, ActionType
from catanatron.models.player import Player


class RandomAgent(Player):
    def decide(self, game: Game, playable_actions: Iterable[Action]) -> Action:
        playable_actions = set(playable_actions)

        playable_actions -= set(maritime_trade_possibilities(game.state, self.color))
        playable_actions -= {Action(self.color, ActionType.BUY_DEVELOPMENT_CARD, None)}

        playable_actions = list(playable_actions)

        if (
            game.state.current_prompt == ActionPrompt.MOVE_ROBBER
            or game.state.current_prompt == ActionPrompt.DISCARD
        ):
            return self.handle_special_prompt(game, playable_actions)
        else:
            action = random.choice(playable_actions)

        return action

    def handle_special_prompt(self, game, playable_actions):
        # Handle actions for prompts like MOVE_ROBBER or DISCARD
        return random.choice(playable_actions) if playable_actions else None
