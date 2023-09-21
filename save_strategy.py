import torch
from torch import nn
import random
import itertools
import math
from collections import Counter
import argparse
import os

from game import *
from train import *

import numpy as np
import json

parser = argparse.ArgumentParser()
parser.add_argument("d1", type=int, default=5, help="Number of dice for player 1")
parser.add_argument("d2", type=int, default=5, help="Number of dice for player 2")
parser.add_argument("--sides", type=int, default=6, help="Number of sides on the dice")

args = parser.parse_args()

public_state_length, public_state_length_per_player, *_ = calculate_arguments(args.d1, args.d2, args.sides)
#model = NetConcat(public_state_length_per_player, public_state_length)
model = NetCompBilin(public_state_length_per_player, public_state_length)
game = Game(args.d1, args.d2, args.sides, model)

def run_simulations():
    all_rolls = list(itertools.product(game.rolls(0), game.rolls(1)))
    for t in range(100):
        replay_buffer = []


        if t % 1 == 0:
            with torch.no_grad():
                roll, strategy = print_strategy_2(game.make_state().to(device))
                data_to_visualise = parse_strategy(strategy)

def print_strategy_2(state):
    total_v = 0
    total_cnt = 0
    for r1, cnt in sorted(Counter(game.rolls(0)).items()):
        priv = game.make_priv(r1, 0).to(device)
        v = model(priv, state)
        rs = torch.tensor(game.make_regrets(priv, state, last_call=-1))
        if rs.sum() != 0:
            rs /= rs.sum()
        strat = []
        for action, prob in enumerate(rs):
            n, d = divmod(action, game.sides)
            n, d = n + 1, d + 1
            if d == 1:
                strat.append(f"{n}:")
            strat.append(f"{prob:.2f}")
        print(r1, f"{float(v):.4f}".rjust(7), f"({cnt})", " ".join(strat))
        total_v += v
        total_cnt += cnt
    print(f"Mean value: {total_v / total_cnt}")
    return r1, strat

run_simulations()