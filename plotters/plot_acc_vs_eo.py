"""
    Retrofit of plot_steps.py to plot accuracy versus EO for various trials.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from adjustText import adjust_text # Hooray for Phyla
import random # ... though their lib does need some adjusting


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

eos = []
accs = []
labels = []

for df, name in zip(file_dfs, names):
    
    min_eo_idx = df[df['is_test'] == 1].filter(regex='_EO').idxmin()
    min_eo_accuracy = df.filter(regex='accuracy').iloc[min_eo_idx]
    min_eo = df.filter(regex='_EO').iloc[min_eo_idx]
    eos.append(min_eo.to_numpy()[0][0])
    accs.append(min_eo_accuracy.to_numpy()[0][0])
    labels.append(name)


fig, ax = plt.subplots()

middle_eo = (max(eos) - min(eos)) / 2.0 + min(eos)


ax.scatter(eos, accs, marker='+', c=colours)

"""

for idx, label in enumerate(labels):
    ax.annotate(label, xy = (eos[idx], accs[idx]), xytext = (0, 10), textcoords = 'offset pixels', ha=('left' if eos[idx] < middle_eo else 'right'), fontsize = 'x-small')
"""

label_texts = []
for idx, label in enumerate(labels):
    label_texts.append(plt.text(eos[idx] + random.random() * 0.0001, accs[idx] + random.random() * 0.0001, label, ha='center', va='center'))

#adjust_text(label_texts)
adjust_text(label_texts, arrowprops=dict(arrowstyle='-', color='black'))
 
ax.set(xlabel = 'minimum equalized odds violation', ylabel='accuracy at minimum EO')

plt.show()


"""
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
np.random.seed(1)
x, y = np.random.random((2,500))

fig, ax = plt.subplots()
ax.plot(x, y, 'bo')
ts = []
for i in range(10):
    ts.append(plt.text(x[i], y[i], 'Something'+str(i)))
adjust_text(ts, x=x, y=y, force_points=0.1, arrowprops=dict(arrowstyle='->', 
color='red'))
plt.show()
"""  
