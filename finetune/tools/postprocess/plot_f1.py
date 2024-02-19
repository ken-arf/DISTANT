import sys
import os
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('f1_result_plot.csv')

print(df)

print(df['fname'].tolist())
new_names = []
for fname in df['fname'].tolist():
    if "annotate.W6" in fname:
        name = "ChunkUP"
    elif "annotate.Scratch" in fname:
        name = "Scratch"
    elif "annotate.It" in fname:
        name = "IterUP"
    else:
        name = "Init"
    new_names.append(name)

print(new_names)

df['name'] = new_names

print(df)

sns.color_palette()

#sns.scatterplot(data=df, x="ratio", y="f1-score", hue="name")

fig, axes = plt.subplots(1, 3)

sns.lineplot(data=df, x="ratio", y="precision", marker='o', markers=True, hue="name", ax=axes[0])
sns.lineplot(data=df, x="ratio", y="recall", marker='o', markers=True, hue="name", ax=axes[1])
sns.lineplot(data=df, x="ratio", y="f1-score", marker='o', markers=True, hue="name", ax=axes[2])


for i in range(3):
    plt.setp(axes[i].get_legend().get_texts(), fontsize='7') # for legend text
    plt.setp(axes[i].get_legend().get_title(), fontsize='8') # for legend title
    axes[i].legend(title='Sim. method')
fig.tight_layout()

plt.show()
