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

    public_state_length = (max_call) * 2 + 1
    public_state_length_per_player = max_call
    n_actions = max_call + 1
    lie_action = max_call
    # cur_index = max_call
    pri_index = max(d1, d2) * sides
    player_info_index = public_state_length - 1

    return public_state_length, public_state_length_per_player, n_actions, lie_action, pri_index, player_info_index

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
            state[self.public_state_length_per_player : self.public_state_length_per_player  * 2]
        )
        player_1_calls =  (player_1_call_range == 1).nonzero(as_tuple=True)[0].tolist()
        return player_0_calls, player_1_calls

    def get_last_call(self, state):
        ids = self.get_calls(state)
        pdb.set_trace()
        if not ids or (len(ids[0]) and len(ids[1])) == 0:
            return -1
        else:
            if(len(ids[0]) > len(ids[1])):
                if(len(ids[0]) == 1):
                    return int(ids[0][0])
                else:
                    return int(ids[0][-1])
            else:
                if(len(ids[1]) == 1):
                    return int(ids[1][0])
                else:
                    return int(ids[1][-1])
    
    def evaluate_call(self, r1, r2, last_call):
        # Players have rolled r1, and r2.
        # Previous actions are `state`
        # Player `caller` just called lie. (This is not included in last_call)
        # Returns True if the call is good, false otherwise

        # Calling lie immediately is an error, so we pretend the
        # last call was good to punish the player.
        if last_call == -1:
            return True

        n, d = divmod(last_call, self.sides)
        n, d = n + 1, d + 1  # (0, 0) means 1 of 1s

        cnt = Counter(r1 + r2)
        actual = cnt[d]

        return actual >= n
    
    def get_legal_calls(self, state):
    # Returns a list of action integers representing legal next moves
        lastCall = self.get_last_call(state)
        legal_actions = []
        for i in range (lastCall + 1, self.lie_action + 1):
            legal_actions.append(i)
        return legal_actions

    def play_random_round(self):
        r1 = random.choice(list(self.rolls(0)))
        r2 = random.choice(list(self.rolls(1)))
        privs = [self.make_priv(r1, 0), self.make_priv(r2, 1)]
        state = self.make_state()

        self.makeRandomMove(state)

    def makeRandomMove(self, state):
        player = self.get_player_turn(state)
        possible_moves = self.get_legal_calls(state)
        selected_move = random.choice(list(possible_moves))
        print(selected_move)


# Utility functions  
def convert_call_to_action_integer(n, d):
    # Of the form such that if you are calling '3 5s', n is 3, d is 5
    # Action ranges from 0 to the maximum call and represents the index in the pytorch tensor that will be set to 1 to represent the call
    action = (n - 1) * 6 + (d - 1)
    return action 

def convert_action_to_call(action):
        # Action is an integer
        for d in range (6):
            d += 1
            n = (action + 7 - d) / 6
            if(n.is_integer()):
                return (n, d)
        return (0, 0)



# For testing purposes:
game = Game(5, 5, 6)
game.play_random_round()
# r1 = random.choice(list(game.rolls(0)))
# r2 = random.choice(list(game.rolls(1)))
# privs = [game.make_priv(r1, 0), game.make_priv(r2, 1)]
# state = game.make_state()

# player = game.get_player_turn(state)
# lastCall = game.get_last_call(state)
# possible_moves = game.get_legal_calls(state)
# selected_move = random.choice(list(possible_moves))
# print(selected_move)

# #game.makeRandomMove(state)
# #game.play_random_round()



















