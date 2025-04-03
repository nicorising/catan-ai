import random
import numpy as np
from collections import defaultdict
from collections.abc import Iterable

from catanatron.game import Game
from catanatron.models.actions import Action, maritime_trade_possibilities
from catanatron.models.player import Player
from catanatron import Color
from catanatron.models.enums import ActionType, ActionPrompt

class TDAgent(Player):
    def __init__(self, color=Color.BLUE, learning_rate=0.1, discount_factor=0.99, epsilon=1, decay_rate=0.8):
        super().__init__(color)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.q_table = defaultdict(lambda: defaultdict(float))

    def decide(self, game: Game, playable_actions: Iterable[Action]):
        if game.state.current_prompt == ActionPrompt.MOVE_ROBBER or game.state.current_prompt == ActionPrompt.DISCARD :
            return self.handle_special_prompt(game, playable_actions)
        else: 
            return self.pick_standard_action(game, playable_actions)

    def pick_standard_action(self, game, playable_actions):
        state = self.get_state(game)

        if random.random() < self.epsilon:
            action = random.choice(playable_actions)
        else:
            action = self.get_best_action(state, playable_actions)

        while 'TRADE' in str(action[1]):  
            action = random.choice(playable_actions)
        
        return action

    def handle_special_prompt(self, game, playable_actions):
        return random.choice(playable_actions) if playable_actions else None

    def get_state(self, game: Game):
        return str(game.state)

    def get_best_action(self, state, playable_actions):
        if state not in self.q_table:
            return random.choice(playable_actions)
        
        action_values = {action: self.q_table[state][action] for action in playable_actions}
        return max(action_values, key=action_values.get)

    def update_q_value(self, state, action, reward, next_state, next_actions):
        best_next_action = self.get_best_action(next_state, next_actions)
        self.q_table[state][action] = self.q_table[state][action] + self.learning_rate * (reward + self.discount_factor * (self.q_table[next_state][best_next_action]) - self.q_table[state][action])

    def calculate_reward(self, game):
        if game.winning_color() == self.color:
            return game.vps_to_win  
        elif game.winning_color() is not None:
            return -game.vps_to_win  
        else:
            key = f"{self.color}_ACTUAL_VICTORY_POINTS"
            current_vps = game.state.player_state.get(key, 0)
            return current_vps - game.vps_to_win


    def train(self, num_episodes=10000):
        TURNS_LIMIT = 1000
        for episode in range(num_episodes):
            game = Game([TDAgent(Color.BLUE), TDAgent(Color.RED)])
            state = self.get_state(game)
            
            while game.winning_color() is None and game.state.num_turns < TURNS_LIMIT:
                actions = game.state.playable_actions
                playable_actions = set(actions)
                playable_actions -= set(maritime_trade_possibilities(game.state, self.color))
                playable_actions -= {Action(self.color, ActionType.BUY_DEVELOPMENT_CARD, None)}

                playable_actions = list(playable_actions)
                action = self.decide(game, playable_actions)
                previous_state = state

                game.execute(action)
                next_state = self.get_state(game)
                reward = self.calculate_reward(game)
                next_actions = game.state.playable_actions
                playable_actions = set(next_actions)
                playable_actions -= set(maritime_trade_possibilities(game.state, self.color))
                playable_actions -= {Action(self.color, ActionType.BUY_DEVELOPMENT_CARD, None)}

                playable_actions = list(playable_actions)

                self.update_q_value(previous_state, action, reward, next_state, playable_actions)
            
            self.epsilon *= self.decay_rate
            
            print(f"Episode {episode + 1}/{num_episodes} completed")
