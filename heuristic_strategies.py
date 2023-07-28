import random
import itertools
import math
from collections import Counter

from game import *

#game.play_random_game()
#game.make_random_move(state)

game = Game(5, 5, 6)
game.game_in_progress = True
game.r1 = random.choice(list(game.rolls(0)))
game.r2 = random.choice(list(game.rolls(1)))
privs = [game.make_priv(game.r1, 0), game.make_priv(game.r2, 1)]
state = game.make_state()

def basic_tactic_1(player, sides):
    if(player == 1):
        dice = game.r1
    else:
        dice = game.r2
# See which number we have the most of
    number_of_ones = dice.count(1)
    highest_frequency_number = 0
    highest_frequency = dice.count(1)
    for i in range(1, sides + 1):
        frequency = number_of_ones + dice.count(i)
        if(frequency > highest_frequency_number):
            highest_frequency_number = i
            highest_frequency = frequency
    print(highest_frequency_number, frequency)
# Estimate how many of this number we expect opponent to have
    if (player == 1):
        num_of_opponent_dice = len(game.r2)
    else:
        num_of_opponent_dice = len(game.r1)
    exp_freq = round(calc_expected_frequency_of_number(highest_frequency_number, num_of_opponent_dice))
    bid_freq = exp_freq + highest_frequency
    bid_num = highest_frequency_number
    bid = (bid_freq, bid_num)
    return bid
    
def calc_expected_frequency_of_number(number, num_dice):
    if(number == 1):
        expected_frequency = num_dice * 1/6
    else:
        expected_frequency= num_dice * 1/3

    return expected_frequency
