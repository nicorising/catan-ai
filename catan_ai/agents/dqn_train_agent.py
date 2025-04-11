import math
import random
from collections import deque, namedtuple
from collections.abc import Iterable

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from catanatron.game import Action, Color, Game, Player, State
from catanatron.models.actions import maritime_trade_possibilities
from catanatron.models.enums import CITY, RESOURCES, ROAD, SETTLEMENT, ActionPrompt, ActionType
from catanatron.models.map import NUM_EDGES, NUM_NODES, NUM_TILES
from catanatron.state_functions import (
    get_player_buildings,
    get_visible_victory_points,
    player_num_resource_cards,
)

GAME_OBSERVATIONS = 2 + len(RESOURCES)
TILE_OBSERVATIONS = NUM_TILES * len(RESOURCES)
N_OBSERVATIONS = GAME_OBSERVATIONS + TILE_OBSERVATIONS + NUM_NODES + NUM_EDGES
N_ACTIONS = NUM_NODES + NUM_EDGES + 1

MEMORY = 100_000
BATCH_SIZE = 10_000
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 400_000.0
LR = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


class CatanDQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()

        self.layer1 = nn.Linear(n_observations, 384)
        self.layer2 = nn.Linear(384, 256)
        self.layer3 = nn.Linear(256, 192)
        self.layer4 = nn.Linear(192, 160)
        self.layer5 = nn.Linear(160, n_actions)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout(F.relu(self.layer2(x)))
        x = self.dropout(F.relu(self.layer3(x)))
        x = self.dropout(F.relu(self.layer4(x)))
        return self.layer5(x)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class DQNTrainAgent(Player):
    def __init__(self, color: Color, path: str | None = None) -> None:
        super().__init__(color)

        self.policy_net = CatanDQN(N_OBSERVATIONS, N_ACTIONS)

        if path is not None:
            state_dict = torch.load(path, map_location=DEVICE)
            self.policy_net.load_state_dict(state_dict)

        self.policy_net = self.policy_net.to(DEVICE)

        self.memory = ReplayMemory(MEMORY)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.steps_done = 0

        self.tile_to_idx: dict[tuple[int, int, int], int] | None = None
        self.edge_to_idx: dict[tuple[int, int], int] | None = None
        self.idx_to_edge: list[tuple[int, int]] | None = None

        self.last_decide = None
        self.losses = []

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)
        self.policy_net.load_state_dict(torch.load(path, map_location=DEVICE))

    def plot_loss(self):
        plt.plot(self.losses)
        plt.title("Loss Over Training")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.show()

    def game_over(self, game: Game) -> None:
        game_result_reward = (
            2 if game.winning_color() == self.color else 0 if game.winning_color() is None else -2
        )
        game_result_reward = 0
        if self.last_decide is not None:
            obs, action_tensor, reward = self.last_decide

            self.memory.push(
                obs,
                action_tensor,
                self.state_to_obs(game.state),
                torch.tensor(reward + game_result_reward, device=DEVICE, dtype=torch.float),
            )

            self.last_decide = None

        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        state_batch, action_batch, next_state_batch, reward_batch = zip(*transitions)

        state_batch = torch.stack(state_batch)
        action_batch = torch.tensor(action_batch, device=DEVICE)
        reward_batch = torch.tensor(reward_batch, device=DEVICE)

        estimates = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()

        next_states = torch.stack(next_state_batch)
        targets = reward_batch.clone()
        targets += GAMMA * torch.max(self.policy_net(next_states), dim=1)[0]

        loss = nn.MSELoss()(estimates, targets)
        self.losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def action_to_index(self, action: Action) -> int:
        if action.action_type == ActionType.END_TURN:
            return 0
        elif action.action_type in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY):
            return action.value + 1
        elif action.action_type == ActionType.BUILD_ROAD:
            edge_idx = self.edge_to_idx[action.value]
            return NUM_NODES + 1 + edge_idx

    def index_to_action(self, index: int, state: State) -> Action:
        if index == 0:
            return Action(self.color, ActionType.END_TURN, None)
        elif 1 <= index <= NUM_NODES:
            if index - 1 in get_player_buildings(state, self.color, SETTLEMENT):
                return Action(self.color, ActionType.BUILD_CITY, index - 1)
            else:
                return Action(self.color, ActionType.BUILD_SETTLEMENT, index - 1)
        else:
            edge_idx = index - (NUM_NODES + 1)
            edge = self.idx_to_edge[edge_idx]
            return Action(self.color, ActionType.BUILD_ROAD, edge)

    def state_to_obs(self, state: State):
        vp = get_visible_victory_points(state, self.color)
        enemy_vp = sum(
            get_visible_victory_points(state, enemy.color)
            for enemy in state.players
            if enemy.color != self.color
        )
        resources = [
            player_num_resource_cards(state, self.color, resource) for resource in RESOURCES
        ]

        game_obs = torch.tensor([vp, enemy_vp, *resources], device=DEVICE, dtype=torch.float)

        num_resources = len(RESOURCES)
        tile_obs = torch.zeros((NUM_TILES, num_resources), device=DEVICE, dtype=torch.float)

        for coordinate, tile in state.board.map.land_tiles.items():
            idx = self.tile_to_idx[coordinate]
            if tile.resource is not None:
                resource_idx = RESOURCES.index(tile.resource)
                tile_obs[idx, resource_idx] = 1

        tile_obs = tile_obs.view(-1)

        node_obs = torch.zeros(NUM_NODES, device=DEVICE, dtype=torch.float)
        edge_obs = torch.zeros(NUM_EDGES, device=DEVICE, dtype=torch.float)

        for player in state.players:
            for node in get_player_buildings(state, player.color, SETTLEMENT):
                node_obs[node] = 1 if player.color == self.color else -1
            for node in get_player_buildings(state, player.color, CITY):
                node_obs[node] = 2 if player.color == self.color else -2
            for edge in get_player_buildings(state, player.color, ROAD):
                edge_obs[self.edge_to_idx[edge]] = 1 if player.color == self.color else -1

        return torch.cat((game_obs, tile_obs, node_obs, edge_obs))

    def play_turn_decide(self, game: Game, playable_actions: Iterable[Action]) -> Action:
        obs = self.state_to_obs(game.state)

        if self.last_decide is not None:
            state, action, reward = self.last_decide

            state = state
            action = action
            next_state = obs
            reward = torch.tensor(reward, device=DEVICE, dtype=torch.float)

            self.memory.push(state, action, next_state, reward)

        valid_mask = torch.full((N_ACTIONS,), -float("inf"), device=DEVICE)
        valid_indices = []
        for action in playable_actions:
            action_idx = self.action_to_index(action)

            valid_mask[action_idx] = 0.0
            valid_indices.append(action_idx)

        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * self.steps_done / EPS_DECAY
        )
        self.steps_done += 1
        self.eps_threshold = eps_threshold

        if random.random() > eps_threshold:
            with torch.no_grad():
                q_values = self.policy_net(obs)
                masked_q_values = q_values + valid_mask
                action_tensor = torch.argmax(masked_q_values)
        else:
            action_idx = random.choice(valid_indices)
            action_tensor = torch.tensor(action_idx, device=DEVICE, dtype=torch.long)

        action = self.index_to_action(action_tensor.item(), game.state)

        if game.state.current_prompt not in [
            ActionPrompt.BUILD_INITIAL_SETTLEMENT,
            ActionPrompt.BUILD_INITIAL_ROAD,
        ]:
            rewards = {
                ActionType.BUILD_SETTLEMENT: 1,
                ActionType.BUILD_CITY: 1,
                ActionType.BUILD_ROAD: 0,
            }
            reward = rewards.get(action.action_type, 0)
        else:
            reward = 0

        self.last_decide = (obs, action_tensor, reward)
        return action

    def decide(self, game: Game, playable_actions: Iterable[Action]) -> Action:
        if self.tile_to_idx is None:
            # Generate the board coordinate to index mapping
            self.tile_to_idx = {}
            for idx, coordinate in enumerate(game.state.board.map.land_tiles):
                self.tile_to_idx[coordinate] = idx

        if self.edge_to_idx is None:
            # Generate the edge maps
            self.edge_to_idx = {}
            self.idx_to_edge = []
            for tile in game.state.board.map.land_tiles.values():
                for edge in tile.edges.values():
                    if edge not in self.edge_to_idx:
                        idx = len(self.edge_to_idx) // 2
                        self.edge_to_idx[edge] = idx
                        self.edge_to_idx[tuple(reversed(edge))] = idx
                        self.idx_to_edge.append(tuple(sorted(edge)))

        buy_development = Action(self.color, ActionType.BUY_DEVELOPMENT_CARD, None)
        bad_actions = set(maritime_trade_possibilities(game.state, self.color)) | {buy_development}
        playable_actions = [action for action in playable_actions if action not in bad_actions]

        if game.state.current_prompt in [
            ActionPrompt.PLAY_TURN,
            ActionPrompt.BUILD_INITIAL_SETTLEMENT,
            ActionPrompt.BUILD_INITIAL_ROAD,
        ]:
            roll_action = Action(self.color, ActionType.ROLL, None)
            if playable_actions == [roll_action]:
                action = roll_action
            else:
                action = self.play_turn_decide(game, playable_actions)
        else:
            action = random.choice(playable_actions)

        return action
