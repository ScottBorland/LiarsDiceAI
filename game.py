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

    public_state_length = (max_call + 1) * 2 + 1
    public_state_length_per_player = max_call + 1
    n_actions = max_call + 1
    lie_action = max_call
    # cur_index = max_call
    pri_index = max(d1, d2) * sides
    player_info_index = public_state_length - 1

    return public_state_length, public_state_length_per_player, n_actions, lie_action, pri_index, player_info_index

class NetConcat(torch.nn.Module):
    def __init__(self, d_pri, d_pub):
        super().__init__()

        hiddens = (500, 400, 300, 200, 100)

        layers = [torch.nn.Linear(d_pri + d_pub, hiddens[0]), torch.nn.ReLU()]
        for size0, size1 in zip(hiddens, hiddens[1:]):
            layers += [torch.nn.Linear(size0, size1), torch.nn.ReLU()]
        layers += [torch.nn.Linear(hiddens[-1], 1), nn.Tanh()]
        self.seq = nn.Sequential(*layers)

    def forward(self, priv, pub):
        if len(priv.shape) == 1:
            joined = torch.cat((priv, pub), dim=0)
        else:
            joined = torch.cat((priv, pub), dim=1)
        return self.seq(joined)

class Game:
    def __init__(self, d1, d2, sides, model="none"):
        # Number of dice that player one and player two start with
        self.d1 = d1
        self.d2 = d2
        # Lists containing dice of each player
        self.r1 = []
        self.r2 = []
        # Reference to neural network being used for training
        self.model = model
        # Number of sides on each die (default is 6)
        self.sides = sides
        # Keeps track of whether game is in progress
        self.game_in_progress = True
        # Set when one player challenges
        self.player_who_called_lie = ""
        # Set when one player wins the game
        self.winner = ""
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

    def sample_action(self, priv, state, last_call, eps):
        pi = self.policy(priv, state, last_call, eps)
        action = next(iter(torch.utils.data.WeightedRandomSampler(pi, num_samples=1)))
        return action + last_call + 1

    def policy(self, priv, state, last_call, eps=0):
        regrets = self.make_regrets(priv, state, last_call)
        #pdb.set_trace()
        if(len(regrets) == 0):
            print(self.lie_action)
            print(priv, state, last_call)
        else:
            for i in range(len(regrets)):
                regrets[i] += eps
            if sum(regrets) <= 0:
                return [1 / len(regrets)] * len(regrets)
            else:
                s = sum(regrets)
                return [r / s for r in regrets]

    def make_regrets(self, priv, state, last_call):
        
        # if priv[self.pri_index] != state[self.player_info_index]:
        #     print("Warning: Regrets are not with respect to current player")
    
        # Number of child nodes
        num_actions = self.n_actions - last_call - 1
        

        # One for the current state, and one for each child
        batch = state.repeat(num_actions + 1, 1)

        for i in range(num_actions):
            self._apply_action(batch[i + 1], i + last_call + 1)

        priv_batch = priv.repeat(num_actions + 1, 1)

        v, *vs = list(self.model(priv_batch, batch))
        return [max(vi - v, 0) for vi in vs]
        # The Hedge method
        # return [math.exp(10*(vi - v)) for vi in vs]

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
        if (action == self.lie_action):
            last_call = self.get_last_call(state)
            self.lie_called(state, last_call)
        return state
    
    def lie_called(self, state, last_call):
        self.game_in_progress = False
        self.player_who_called_lie = 1 - self.get_player_turn(state)
        other_player = self.get_player_turn(state)
        if(self.evaluate_call(self.r1, self.r2, last_call)):
            #print("The last bid was true! Player " + str(self.player_who_called_lie) + " wins!")
            self.winner = self.player_who_called_lie
        else:
            #print("The last bid was false! Player " + str(other_player) + " wins!")
            self.winner = other_player
    
    def get_calls(self, state):
        player_0_call_range = (
            state[: self.public_state_length_per_player]
        )
        player_0_calls =  (player_0_call_range == 1).nonzero(as_tuple=True)[0].tolist()
        player_1_call_range = (
            state[self.public_state_length_per_player : self.public_state_length_per_player  * 2]
        )
        player_1_calls =  (player_1_call_range == 1).nonzero(as_tuple=True)[0].tolist()
        #pdb.set_trace()
        if(len(player_0_calls) == 0):
            player_0_calls.append(-1)
        if (len(player_1_calls) == 0):
            player_1_calls.append(-1)
        return player_0_calls, player_1_calls

    def get_last_call(self, state):
        ids = self.get_calls(state)
        #pdb.set_trace()
        if not ids:
            #or (len(ids[0]) and len(ids[1]))
            return -1
        if (len(ids[0]) == 0):
            return -1
        else:
            if(len(ids[0]) > len(ids[1])) or (ids[1][-1] == -1):
                return int(ids[0][-1])
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
        actual = cnt[d] + cnt[1] if d != 1 else cnt[d]
        return actual >= n
    
    def get_legal_calls(self, state):
    # Returns a list of action integers representing legal next moves
        lastCall = self.get_last_call(state)
        legal_actions = []
        for i in range (lastCall + 1, self.lie_action + 1):
            legal_actions.append(i)
        return legal_actions

    def play_random_game(self):
        self.game_in_progress = True
        self.r1 = random.choice(list(self.rolls(0)))
        self.r2 = random.choice(list(self.rolls(1)))
        privs = [self.make_priv(self.r1, 0), self.make_priv(self.r2, 1)]
        state = self.make_state()
        while self.game_in_progress == True:
            state = self.make_random_move(state)
        calls = self.get_calls(state)
        self.show_game_information(self.r1, self.r2, calls)
    
    def show_game_information(self, r1, r2, calls):
        # Print out the dice of each player to the console along with a list of the calls made throughout the game and the final winner
        print("Player 1 Dice: " + str(r1))
        print("Player 2 Dice: " + str(r2))
        player0callsList = []
        player1callsList = []
        for action in calls[0]:
            call = convert_action_to_call(action)
            player0callsList.append(call)
        for action in calls[1]:
            call = convert_action_to_call(action)
            player1callsList.append(call)
        turn = 0
    
        for x in range(len(player0callsList)):
            if(turn == 0):
                print("Player 1 bids: "  + str(player0callsList[x][0]) + " " + str(player0callsList[x][1]) + "s")
            else:
                if((x + 1) > len(player1callsList)):
                    print('break')
                    break
                else:
                    print("Player 2 bids: "  + str(player1callsList[x][0]) + " " + str(player1callsList[x][1]) + "s")
            turn = 1 - turn

        print("Player " + str(self.player_who_called_lie) +  " calls lie!")
        if(self.player_who_called_lie == self.winner):
            print("The last bid was true! Player " + str(self.player_who_called_lie + 1) + " wins!")
        else:
            otherPlayer = 1 - int(self.player_who_called_lie)
            print("The last bid was false! Player " + str(otherPlayer + 1) + " wins!")

    def make_random_move(self, state):
        player = self.get_player_turn(state)
        possible_moves = self.get_legal_calls(state)
        selected_move = random.choice(list(possible_moves))
        # print("Selected move: " + str(selected_move))
        # print("Player : " + str(player))
        state = self.apply_action(state, selected_move)
        return state

# Utility functions  
def convert_call_to_action_integer(n, d):
    # Of the form such that if you are calling '3 5s', n is 3, d is 5
    # Action ranges from 0 to the maximum call and represents the index in the pytorch tensor that will be set to 1 to represent the call
    action = (n - 1) * 6 + (d - 1)
    return action 

def convert_action_to_call(action):
        # Action is an integer
        # Old method
        # call = [0, 0]
        # for d in range (6):
        #     d += 1
        #     n = (action + 7 - d) / 6
        #     if(n.is_integer()):
        #         call[0] = int(n)
        #         call[1] = d
        # return call

        # New method
        n, d = divmod(action)
        call = (n + 1, d + 1)
        return(call)

# For testing purposes:
game = Game(5, 5, 6)
#pdb.set_trace()
#game.play_random_game()



game.game_in_progress = True
game.r1 = random.choice(list(game.rolls(0)))
game.r2 = random.choice(list(game.rolls(1)))
privs = [game.make_priv(game.r1, 0), game.make_priv(game.r2, 1)]
state = game.make_state()

#pdb.set_trace()

# state = game.make_random_move(state)
# state = game.apply_action(state, 6)
# lieAction = game.lie_action
# state = game.apply_action(state, lieAction)
# calls = game.get_calls(state)
# print(calls)
# pdb.set_trace()
# state = game.apply_action(state, 0)
# legalActions2 = game.get_legal_calls(state)
# print(legalActions2)
# state = game.apply_action(state, 18)
# legalActions3 = game.get_legal_calls(state)
# print(legalActions3)
# state = game.apply_action(state, 58)
# legalActions4 = game.get_legal_calls(state)
# print(legalActions4)
# state = game.apply_action(state, 59)
# legalActions5 = game.get_legal_calls(state)
# print(legalActions5)


















