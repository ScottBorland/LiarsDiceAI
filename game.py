import pdb
import random
import torch
from torch import nn
import itertools
import numpy as np
import math
from collections import Counter

def calculate_arguments(d1, d2, sides):
    max_call = (d1 + d2) * sides

    public_state_length = (max_call + 1) * 2
    public_state_length_per_player = max(d1, d2) * sides + 1
    n_actions = max_call + 1
    lie_action = max_call
    # cur_index = max_call
    pri_index = max(d1, d2) * sides
    player_info_index = public_state_length - 1
    D_PUB_PER_PLAYER = max_call + 1

    return public_state_length, public_state_length_per_player, n_actions, lie_action, pri_index, player_info_index, D_PUB_PER_PLAYER

class Game:
    def __init__(self, d1, d2, sides):
        # Number of dice that player one and player two start with
        self.d1 = d1
        self.d2 = d2
        # Number of sides on each die (default is 6)
        self.sides = sides
        (
            # Length of tensor needed to represent the public game state which is a pytorch tensor representing calls that have been made
            self.public_state_length,
            # Length of tensor needed to represent the calls made by one player, and whether it is their turn
            self.public_state_length_per_player,
            # Number of possible actions where each action is represented as an integer (or 'lie')
            self.n_actions,
            # Integer which is used to represent the action 'lie', representing one player challenging the previous call
            self.lie_action,
            # This is the index in the private tensor which indicates whether these are the dice or player 0 or player 1
            self.pri_index,
            # This is the index in state which indicates whose turn it is
            self.player_info_index,
            # Same as public_state_length_per_player (will figure out why separate variable later)
            self.D_PUB_PER_PLAYER,
        ) = calculate_arguments(d1, d2, sides)

    def make_priv(self, roll, player):
        # roll is a list of integers
        assert player in [0, 1]
        priv = torch.zeros(self.public_state_length_per_player)
        # Final node in priv tensor represents whether this is the hand of player 0 or 1
        priv[self.pri_index] = player
        # New method inspired by Chinese poker paper for representing the player's dice in a "6 × 'number of dice' one-hot representation"
        cnt = Counter(roll)
        for face, c in cnt.items():
            for i in range(c):
                priv[(face - 1) * max(self.d1, self.d1) + i] = 1    
        return priv

    def make_state(self):
        state = torch.zeros(self.public_state_length)
        state[self.player_info_index] = 0
        return state
    
    def rolls(self, player):
        assert player in [0, 1]
        n_dice = self.d1 if player == 0 else self.d2
        # Generates array of all possible arrays of dice rolls for n_dice
        return [
            tuple(sorted(r))
            for r in itertools.product(range(1, self.sides + 1), repeat=n_dice)
        ]
    
    def get_player_turn(self, state):
        # Whose turn is it?
        return int(state[self.player_info_index])
    
    def apply_action(self, state, action):
        new_state = state.clone()
        self._apply_action(new_state, action)
        return new_state

    def _apply_action(self, state, action):
        player_next_to_act = self.get_player_turn(state)
        state[action + player_next_to_act * self.public_state_length_per_player] = 1
        state[self.player_info_index] = 1 - state[self.player_info_index]
        return state
    
    def get_calls(self, state):
        player_0_call_range = (
            state[: self.public_state_length_per_player]
        )
        player_0_calls =  (player_0_call_range == 1).nonzero(as_tuple=True)[0].tolist()
        player_1_call_range = (
            state[self.public_state_length_per_player : self.lie_action]
        )
        player_1_calls =  (player_1_call_range == 1).nonzero(as_tuple=True)[0].tolist()
        return player_0_calls, player_1_calls

    # def get_last_call(self, state):
    #     ids = self.get_calls(state)
    #     if not ids:
    #         return -1
    #     return int(ids[-1])

# For testing purposes:
game = Game(5, 5, 6)

roll1 = game.rolls(0)[23]
roll2 = game.rolls(1)[43]

game.make_priv(roll1, 0)
game.make_priv(roll2, 1)

state = game.make_state()
state = game.apply_action(state, 29)
state = game.apply_action(state, 2)
state = game.apply_action(state, 15)
state = game.apply_action(state, 22)
calls = game.get_calls(state)

pdb.set_trace()



















