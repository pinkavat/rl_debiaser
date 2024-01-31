"""
    Parse command-line args as csv files; filter for 'EO', then graph together.
    Updated from plot_eo.py to handle per-item measurements
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


file_dfs = []
names = []
for filename in sys.argv[1:]:
    try:
        if not ("critic_loss" in filename): # Don't want to accidentally track up critic loss
            x = pd.read_csv(filename)
            file_dfs.append(x)
            names.append(filename.split('/')[-1][:-4])
    except:
        print(f"Couldn't open {filename}")


colours = plt.cm.Set1(np.linspace(0, 1, len(file_dfs)))
alt_colours = [[c[0],c[1],c[2], 0.4] for c in colours]

labelled_lines = []
labels = []

for index, df, name in zip(range(len(file_dfs)), file_dfs, names):
    grouped = df.groupby('episode')
    for group in grouped.groups:
        
        # Plot an episode's worth of steps

        x = grouped.get_group(group)['substep'].to_numpy()
        y = grouped.get_group(group).filter(regex='_EO').to_numpy()
        
        plt.plot(x, y, color = alt_colours[index])

    # Plot the episode-ending test steps
    test_steps = df[df['is_test'] == 1]
    x = test_steps['substep'].to_numpy()
    y = test_steps.filter(regex='_EO').to_numpy()

    epoch_line = plt.plot(x, y, color = colours[index], marker='.')[0]

    labelled_lines.append(epoch_line)
    labels.append(name)

plt.legend(labelled_lines, labels)

plt.xlabel('substep')
plt.ylabel('binary Equalized Odds maximum violation')

plt.show()

#full_df = pd.concat(file_dfs, axis=1)
#full_df = full_df.loc[:,~full_df.columns.duplicated()].copy() # Deduplication of columns, after https://stackoverflow.com/questions/14984119/python-pandas-remove-duplicate-columns


#fairnesses = full_df.filter(regex='EO')


#fig, ax = plt.subplots()

#ax.plot()


#ax = fairnesses.plot.line(color=colours)
#ax.xaxis.set_major_locator(MaxNLocator(integer=True))


#plt.xlabel('Step')
#plt.ylabel('Binary Equalized Odds: Maximum Violation')
#plt.show()
