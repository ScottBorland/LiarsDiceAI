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
    cur_index = max_call
    pri_index = max(d1, d2) * sides
    D_PUB_PER_PLAYER = max_call + 1

    return public_state_length, public_state_length_per_player, n_actions, lie_action, cur_index, pri_index, D_PUB_PER_PLAYER

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
            # These indicate the halfway point of the public game state tensor which is where info is given about from who's perspective we are given the information
            self.cur_index,
            self.pri_index,
            # Same as public_state_length_per_player (will figure out why separate variable later)
            self.D_PUB_PER_PLAYER,
        ) = calculate_arguments(d1, d2, sides)

    def make_priv(self, roll, player):
        # roll is a list of integers
        assert player in [0, 1]
        priv = torch.zeros(self.public_state_length_per_player)
        priv[self.pri_index] = player
        # New method inspired by Chinese poker paper
        cnt = Counter(roll)
        for face, c in cnt.items():
            for i in range(c):
                priv[(face - 1) * max(self.d1, self.d1) + i] = 1    
        return priv

    def make_state(self):
        state = torch.zeros(self.public_state_length)
        state[self.cur_index] = 1
        pdb.set_trace()
        return state

game = Game(5, 5, 6)
roll = [1, 2, 4, 4]
game.make_priv(roll, 1)
game.make_state()


















