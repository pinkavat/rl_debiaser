"""
    Parse command-line args as csv files; filter for 'EO', then graph together.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

file_dfs = []
serieses = 0
for filename in sys.argv[1:]:
    try:
        x = pd.read_csv(filename)
        file_dfs.append(x)
        serieses += 1
    except:
        print(f"Couldn't open {filename}")

full_df = pd.concat(file_dfs, axis=1)
fairnesses = full_df.filter(regex='EO')

colours = plt.cm.Set1(np.linspace(0, 1, serieses))
ax = fairnesses.plot.line(color=colours)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))


plt.xlabel('Episode')
plt.ylabel('Binary Equalized Odds: Maximum Violation')
plt.show()
