import random
from collections.abc import Iterable

from catanatron.game import Game
from catanatron.models.actions import Action
from catanatron.models.player import Player


class RandomAgent(Player):
    def decide(self, game: Game, playable_actions: Iterable[Action]):
        print(playable_actions)
        return random.choice(playable_actions)
