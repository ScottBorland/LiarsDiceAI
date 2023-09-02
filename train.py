import torch
from torch import nn
import random
import itertools
import math
from collections import Counter
import argparse
import os

from game import *

import numpy as np
import json

parser = argparse.ArgumentParser()
parser.add_argument("d1", type=int, default=5, help="Number of dice for player 1")
parser.add_argument("d2", type=int, default=5, help="Number of dice for player 2")
parser.add_argument("--sides", type=int, default=6, help="Number of sides on the dice")
parser.add_argument(
    "--eps", type=float, default=1e-2, help="Added to regrets for exploration"
)
parser.add_argument(
    "--layers", type=int, default=4, help="Number of fully connected layers"
)
parser.add_argument(
    "--layer-size", type=int, default=100, help="Number of neurons per layer"
)
parser.add_argument("--lr", type=float, default=1e-3, help="LR = lr/t")
parser.add_argument("--w", type=float, default=1e-2, help="weight decay")
parser.add_argument(
    "--path", type=str, default="models/model5v5_2", help="Where to save checkpoints"
)

args = parser.parse_args()

# Check if there is a model we should continue training
if os.path.isfile(args.path):
    checkpoint = torch.load(args.path)
    print(f"Using args from {args.path}")
    old_path = args.path
    args = checkpoint["args"]
    args.path = old_path
else:
    checkpoint = None

# Model : (private state, public state) -> value
public_state_length, public_state_length_per_player, *_ = calculate_arguments(args.d1, args.d2, args.sides)
model = NetConcat(public_state_length_per_player, public_state_length)
# model = Net(D_PRI, D_PUB)
# model = Net2(D_PRI, D_PUB)
game = Game(args.d1, args.d2, args.sides, model)

if checkpoint is not None:
    print("Loading previous model for continued training")
    model.load_state_dict(checkpoint["model_state_dict"])


device = torch.device("cpu")
model.to(device)

p1_WIN = 1
p1_LOSE = -1

@torch.no_grad()
def play(r1, r2, replay_buffer):
    privs = [game.make_priv(r1, 0).to(device), game.make_priv(r2, 1).to(device)]

    def play_inner(state):
        current_player = game.get_player_turn(state)
        calls = game.get_calls(state)
      
        def lie_called_on_first_action(calls):
            if calls[0][-1] == game.lie_action and len(calls[0]) == 1:
                return True
            else: return False
        
        def lie_called(calls):
            if calls[0][-1] or calls[1][-1] == game.lie_action:
                return True
            else: return False

        if lie_called_on_first_action(calls): res = p1_LOSE
        elif lie_called(calls):
            if current_player == 0:
                prev_call = game.get_last_call(state)
            else:
                prev_call = game.get_last_call(state)
            res = p1_WIN if game.evaluate_call(r1, r2, prev_call) else p1_LOSE
        else:
            if current_player == 0:
                last_call = game.get_last_call(state)
            else:
                last_call = game.get_last_call(state)

            action = game.sample_action(privs[current_player], state, last_call, args.eps)
            new_state = game.apply_action(state, action)
            # Repeat function until game is concluded
            res = -play_inner(new_state)
        
        # Save the result from the perspective of both sides
        replay_buffer.append((privs[current_player], state, res))
        replay_buffer.append((privs[1 - current_player], state, -res))

        return res

    with torch.no_grad():
        state = game.make_state().to(device)
        play_inner(state)

def print_strategy(state):
    total_v = 0
    total_cnt = 0
    strats = []
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
        strat.append(r1)
        strats.append(strat)
    print(f"Mean value: {total_v / total_cnt}")
    return strats

def write_to_json(strat, roll):
    strat['roll'] = roll
    # Writing to strategy.json
    with open("strategy.json", "r+") as file:
        file_data = json.load(file)
        file_data["strategy"].append(strat)
        file.seek(0)
        json.dump(file_data, file, indent = 4)

class ReciLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, gamma=1, last_epoch=-1, verbose=False):
        self.gamma = gamma
        super(ReciLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [
            base_lr / (self.last_epoch + 1) ** self.gamma
            for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
        ]

    def _get_closed_form_lr(self):
        return [
            base_lr / (self.last_epoch + 1) ** self.gamma for base_lr in self.base_lrs
        ]
    
def parse_strategy(strategies):
    for strategy in strategies:
        roll = strategy.pop()
        result_dictionary = {}
        key = '-1'
        for i, item in enumerate(strategy):
            if ":" in item:
                if key != '-1':
                    result_dictionary[key] = probabilities
                key = item
                probabilities = []
            else:
                probabilities.append(item)
                if i == (len(strategy) - 1):
                    result_dictionary[key] = probabilities
        write_to_json(result_dictionary, roll)      
    return result_dictionary, roll

def train():
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.w)
    scheduler = ReciLR(optimizer, gamma=0.5)
    value_loss = torch.nn.MSELoss()
    all_rolls = list(itertools.product(game.rolls(0), game.rolls(1)))
    for t in range(30000):
        replay_buffer = []

        BS = 100  # Number of rolls to include
        for r1, r2 in (
            all_rolls if len(all_rolls) <= BS else random.sample(all_rolls, BS)
        ):
            play(r1, r2, replay_buffer)

        random.shuffle(replay_buffer)
        privs, states, y = zip(*replay_buffer)

        privs = torch.vstack(privs).to(device)
        states = torch.vstack(states).to(device)
        y = torch.tensor(y, dtype=torch.float).reshape(-1, 1).to(device)

        y_pred = model(privs, states)

        # Compute and print loss
        loss = value_loss(y_pred, y)
        print(t, loss.item())

        if t % 10 == 0:
            with torch.no_grad():
                strategy = print_strategy(game.make_state().to(device))
                data_to_visualise = parse_strategy(strategy)
                # visualise_strategy(data_to_visualise, roll)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (t + 1) % 200 == 0:
            print(f"Saving to {args.path}")
            torch.save(
                {
                    "epoch": t,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": args,
                },
                args.path,
            )
        if (t + 1) % 1000 == 0:
            torch.save(
                {
                    "epoch": t,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": args,
                },
                f"{args.path}.cp{t+1}",
            )

train()
