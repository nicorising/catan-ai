import random
from collections.abc import Iterable

from catanatron import ActionType
from catanatron.game import Game
from catanatron.models.actions import (
    Action,
    maritime_trade_possibilities,
)
from catanatron.models.player import Color, Player
from catanatron.state_functions import get_actual_victory_points


class Node(object):
    def __init__(self, col: Color, act: Action):
        """
        Initializes a node.
        :param game: The game we're playing in.
        """
        self.first = True
        self.color = col
        self.wins = 0
        self.action = act
        self.children = []

    def add_child(self, child: "Node"):
        self.children.append(child)

    def simulate(self, game: Game, agent: "MCTS", sims):
        """
        Simluates a specified amount of games and returns stats.
        :param game: The game to take simulations of.
        :param agent: The agent that called this simulation.
        :param sims: Amount of simulations.
        :return: Amount of wins, average victory points for losses and ties, amount of sims.
        """
        # create a copy of the game
        gc = game.copy()
        # execute our action
        gc.play_tick(decide_fn=lambda player, gc, actions: self.action)
        # play some amount of games, taking some random action
        wins = 0
        # track average amount of victory points when we lose/run out of turns
        avg_vp = 0
        nonwins = 0
        for i in range(
            sims
        ):
            # do random action
            if gc.play(decide_fn=lambda player, gc, actions: random.choice(actions)) == self.color:
                wins += 1
            else:
                nonwins += 1
                # add to running avg
                avg_vp += (get_actual_victory_points(gc.state, self.color) - avg_vp) / nonwins

        return wins, avg_vp, sims


class MCTS(Player):
    def __init__(self, color, is_bot=True):
        """
        Initializes a new MCTS player, creating the initial tree node.
        Each node in this implementation corresponds to a specific action, rather than
        a reachable game state. It would be difficult to account for all game states
        given the randomness inherent to Catan.
        :param color: This player's color.
        """
        super().__init__(color, is_bot)
        # are we currently simulating the game? need to know for when decide is called
        self.node = None

    def evaluate(self, wins, vp_avg, sims):
        win_rate = wins / sims
        return win_rate + 0.5 * vp_avg

    def decide(self, game: Game, playable_actions: Iterable[Action]):
        """
        Decide which action to take based on a simulation of games that occur
        after choosing a certain action.
        :param game: The game to take an action in.
        :param playable_actions: The actions to choose from.
        :return: The optimal action to take via highest wins %.
        """
        # adjust actions
        playable_actions = set(playable_actions)
        playable_actions -= set(maritime_trade_possibilities(game.state, self.color))
        playable_actions -= {Action(self.color, ActionType.BUY_DEVELOPMENT_CARD, None)}
        playable_actions = list(playable_actions)

        # create children
        best = (0, None)
        for act in playable_actions:
            # initialize children
            n = Node(self.color, act)
            # simulate the new child's action
            # this runtime can get very high, so keep sims low
            wins, vp_loss_avg, sims = n.simulate(game.copy(), self, 2)
            heuristic = self.evaluate(wins, vp_loss_avg, sims)
            if heuristic > best[0]:
                best = (heuristic, act)
        # return the best action
        return best[1]
