import sys
import os
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


metric = sys.argv[1]

df = pd.read_csv('f1_result_plot.csv')

print(df)

print(df['fname'].tolist())
new_names = []
for fname in df['fname'].tolist():
    if "annotate.W6" in fname:
        name = "ChunkUP"
        name = "ChunkUP"
    elif "annotate.Scratch" in fname:
        name = "Scratch"
    elif "annotate.It" in fname:
        name = "IterUP"
        name = "IterUP"
    else:
        name = "Init"
    new_names.append(name)

print(new_names)

df['name'] = new_names

#print(df)

plt.figure(figsize=(3.5, 3.8))

sns.color_palette()

#sns.scatterplot(data=df, x="ratio", y="f1-score", hue="name")

#fig, axes = plt.subplots(1, 3)

hue_order = ['Init', 'ChunkUP', 'IterUP', 'Scratch']
palette = ['black', 'red', 'blue', 'grey']

#sns.lineplot(data=df, x="ratio", y="precision", marker='o', markers=True, 
#                hue="name", ax=axes[0], hue_order=hue_order, palette=palette)
#sns.lineplot(data=df, x="ratio", y="recall", marker='o', markers=True, 
#                hue="name", ax=axes[1], hue_order=hue_order, palette=palette)
#sns.lineplot(data=df, x="ratio", y="f1-score", marker='o', markers=True, 
#                hue="name", ax=axes[2], hue_order=hue_order, palette=palette)


sns.set_style("darkgrid", {'grid.linestyle': '--'})
with sns.axes_style("darkgrid"):

    sns.lineplot(data=df, x="ratio", y=f"{metric}", hue="name", hue_order=hue_order, palette=palette)
    ax = sns.scatterplot(data=df, x="ratio", y=f"{metric}", hue="name", hue_order=hue_order, palette=palette, legend=False)

    
#sns.legend(title='Sim. method', fontsize='13')
ax.set_xlabel('gold sampling ratio', fontsize='13')
ax.set_ylabel(f'{metric}', fontsize='13')
#plt.legend(title="Sim. method", loc='lower right', fontsize='13')
plt.legend( loc='lower right', fontsize='13')


plt.tight_layout()
plt.savefig(f"{metric}.png")
plt.show()
