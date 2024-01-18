"""
    Parse command-line args as csv files; filter for 'fairness', then graph together.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

file_dfs = []
serieses = 0
series_names = []
for filename in sys.argv[1:]:
    try:
        x = pd.read_csv(filename)
        file_dfs.append(x)
        serieses += 1
        series_names.append(filename[:-4])
    except:
        print(f"Couldn't open {filename}")


colours = plt.cm.Set1(np.linspace(0, 1, serieses))

i = 0
for df in file_dfs:
    colour = colours[i]
    faded_colour = [colour[0], colour[1], colour[2], 0.5]
    plt.fill_between(df.iloc[:, 0].to_numpy(), np.squeeze(df.filter(like='min_reward').to_numpy()) , np.squeeze(df.filter(like='max_reward').to_numpy()), color = faded_colour)
    plt.plot(df.iloc[:, 0].to_numpy(), np.squeeze(df.filter(like='mean_reward').to_numpy()), color = colour, label = series_names[i])
    i += 1

full_df = pd.concat(file_dfs, axis=1)

plt.xlabel('Episode')
plt.ylabel('Reward ($\Delta$ Acceptance Equality)')

plt.legend()
plt.show()
