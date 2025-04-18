import random
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
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
print(N_OBSERVATIONS)
print(N_ACTIONS)


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


class DQNAgent(Player):
    def __init__(self, color: Color, path: str) -> None:
        super().__init__(color)

        state_dict = torch.load(path, map_location=DEVICE)

        self.policy_net = CatanDQN(N_OBSERVATIONS, N_ACTIONS)
        self.policy_net.load_state_dict(state_dict)
        self.policy_net = self.policy_net.to(DEVICE)

        self.tile_to_idx: dict[tuple[int, int, int], int] | None = None
        self.edge_to_idx = None

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

        valid_mask = torch.full((N_ACTIONS,), -float("inf"), device=DEVICE)
        valid_indices = []
        for act in playable_actions:
            act_idx = self.action_to_index(act)
            valid_mask[act_idx] = 0.0
            valid_indices.append(act_idx)

        with torch.no_grad():
            q_values = self.policy_net(obs).squeeze(0)
            masked_q_values = q_values + valid_mask
            action_tensor = torch.argmax(masked_q_values).view(1)

        action = self.index_to_action(action_tensor.item(), game.state)
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
