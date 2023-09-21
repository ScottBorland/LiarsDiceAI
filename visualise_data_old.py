

import numpy as np
import matplotlib.pyplot as plt
import json
import random

# Opening JSON file
with open('strategy.json', 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)

all_data = []

for item in json_object["strategy"]:
    if isinstance(item, dict):
        all_data.append(item)


current_data = all_data[random.randint(1, 200)]
starting_dice = current_data.popitem()

lie_probability = current_data.popitem()


heatmap_data = []
for key in current_data:
    row_data = [float(val) for val in current_data[key]]
    heatmap_data.append(row_data)

# Convert data to a numpy array
graph_data = np.array(heatmap_data, dtype=float)

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the heatmap with inverted y-axis
im = ax.imshow(graph_data, origin='upper', cmap='PuBu')  # Set origin to upper for inverted y-axis

# Set title of graph to starting dice
lie_probability_as_string = str(lie_probability[1])
starting_dice_list = starting_dice[1]
starting_dice_as_string = ' '.join(map(str, starting_dice_list))
ax.set_title('Starting dice: ' + starting_dice_as_string + '      Call_lie probability: ' + lie_probability_as_string)

# Invert the y-axis ticks
ax.invert_yaxis()

# Adjust y-axis ticks and labels to start from 1
num_rows = graph_data.shape[0]
ax.set_yticks(np.arange(num_rows))
ax.set_yticklabels(np.arange(1, num_rows + 1))

# Adjust x-axis ticks and labels to start from 1
num_cols = graph_data.shape[1]
ax.set_xticks(np.arange(num_cols))
ax.set_xticklabels(np.arange(1, num_cols + 1))

# Show the colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Probability", rotation=-90, va="bottom")  # Set colorbar label

plt.show()