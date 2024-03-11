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

sns.color_palette()

#sns.scatterplot(data=df, x="ratio", y="f1-score", hue="name")

fig, axes = plt.subplots(1, 3)

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

    sns.lineplot(data=df, x="ratio", y="precision", hue="name", ax=axes[0], hue_order=hue_order, palette=palette)
    sns.lineplot(data=df, x="ratio", y="recall",    hue="name", ax=axes[1], hue_order=hue_order, palette=palette)
    sns.lineplot(data=df, x="ratio", y="f1-score",  hue="name", ax=axes[2], hue_order=hue_order, palette=palette)

    sns.scatterplot(data=df, x="ratio", y="precision", hue="name", ax=axes[0], hue_order=hue_order, palette=palette, legend=False)
    sns.scatterplot(data=df, x="ratio", y="recall",    hue="name", ax=axes[1], hue_order=hue_order, palette=palette, legend=False)
    sns.scatterplot(data=df, x="ratio", y="f1-score",  hue="name", ax=axes[2], hue_order=hue_order, palette=palette, legend=False)

for i in range(3):
    #plt.setp(axes[i].get_legend().get_texts(), fontsize='12') # for legend text
    #plt.setp(axes[i].get_legend().get_title(), fontsize='12') # for legend title
    
    #axes[i].legend(title='Sim. method', fontsize='13')
    axes[i].legend(fontsize='13')
    axes[i].set_xlabel('gold sampling ratio', fontsize='13')

    # change order of lenged
    #handles, labels = plt.gca().get_legend_handles_labels()
    #print(i, labels)
    #order = [0,1,3,2]
    #axes[i].legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    #axes[i].legend(label_order=order)
    axes[i].grid(linestyle='--')

axes[0].set_ylabel('Precision', fontsize='13')
axes[1].set_ylabel('Recall', fontsize='13')
axes[2].set_ylabel('F1-score', fontsize='13')

fig.tight_layout()

plt.show()
