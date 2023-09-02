import random
import torch
from torch import nn

import os
import sys

from game import *

class Robot:
    def __init__(self, priv, game):
        self.priv = priv
        self.game = game

    def get_action(self, state):
        last_call = self.game.get_last_call(state)
        return self.game.sample_action(self.priv, state, last_call, eps=0)

def load_models(path_1, path_2):

    checkpoint_1 = torch.load(path_1, map_location=torch.device("cpu"))
    args_1 = checkpoint_1["args"]

    public_state_length, public_state_length_per_player, *_ = calculate_arguments(
        args_1.d1, args_1.d2, args_1.sides
    )

    model_1 = NetConcat(public_state_length_per_player, public_state_length)
    model_1.load_state_dict(checkpoint_1["model_state_dict"])

    checkpoint_2 = torch.load(path_2, map_location=torch.device("cpu"))
    args_2 = checkpoint_2["args"]

    model_2 = NetConcat(public_state_length_per_player, public_state_length)
    model_2.load_state_dict(checkpoint_2["model_state_dict"])

    return model_1, model_2

def run_game(game1, game2):
    games = [game1, game2]
    game = games[
        0
    ]  # Just use the first model for the common things, like rolling and stuff
    r1 = random.choice(list(game.rolls(0)))
    r2 = random.choice(list(game.rolls(1)))
    priv1 = game.make_priv(r1, 0)
    priv2 = game.make_priv(r2, 1)
    scores = [0, 0]
    for flip in range(2):
        if not flip:
            players = [Robot(priv1, games[0]), Robot(priv2, games[1])]
        else:
            players = [Robot(priv1, games[1]), Robot(priv2, games[0])]
        state = game.make_state()
        cur = 0
        while True:
            action = players[cur].get_action(state)
            if action == game.lie_action:
                last_call = game.get_last_call(state)
                res = game.evaluate_call(r1, r2, last_call)
                winner = 1 - cur if res else cur
                if not flip:
                    scores[winner] += 1
                else:
                    scores[1 - winner] += 1
                break
            state = game.apply_action(state, action)
            cur = 1 - cur
    return scores 

model_1, model_2 = load_models("C:/Users/Scott/documents/liarsdice/main/models/model5v5_2.pt.cp10000", "C:/Users/Scott/documents/liarsdice/main/models/model5v5_2.pt.cp10000")

game_1 = Game(5, 5, 6, model_1)
game_2 = Game(5, 5, 6, model_2)

total_scores = [0, 0]
for i in range(1000):
    s0, s1 = run_game(game_1, game_2)
    total_scores[0] += s0
    total_scores[1] += s1
pdb.set_trace()