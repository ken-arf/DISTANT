import sys
import os
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns



metric = "f1-score"

#df = pd.read_csv('f1_result_plot.csv')
df = pd.read_csv('f1_result.csv')

sim_cnt = [int(v*10) for v in df['ratio'].tolist()]

df["sim_cnt"] = sim_cnt

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


df_sel = df[df['name'].isin(['Scratch', 'IterUP'])]
max_f1 = df_sel['f1-score'].max()
df_sel['f1-score_rel'] = df_sel['f1-score']/max_f1

print(df_sel)



plt.figure(figsize=(3.5, 3.8))

#sns.color_palette()
sns.color_palette("coolwarm", as_cmap=True)

#sns.scatterplot(data=df, x="ratio", y="f1-score", hue="name")

#fig, axes = plt.subplots(1, 3)

hue_order = [ 'IterUP', 'Scratch']
palette = ['black', 'red', 'blue', 'grey']

#sns.lineplot(data=df, x="ratio", y="precision", marker='o', markers=True, 
#                hue="name", ax=axes[0], hue_order=hue_order, palette=palette)
#sns.lineplot(data=df, x="ratio", y="recall", marker='o', markers=True, 
#                hue="name", ax=axes[1], hue_order=hue_order, palette=palette)
#sns.lineplot(data=df, x="ratio", y="f1-score", marker='o', markers=True, 
#                hue="name", ax=axes[2], hue_order=hue_order, palette=palette)


#sns.set_style("darkgrid", {'grid.linestyle': '--'})
x_tick = [0,1,2,3,4,5,6,7,8,9]
y_tick = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
x_labels = [(x+1)/10.0 for x in x_tick]

metric = 'f1-score_rel'

with sns.axes_style("whitegrid", {'grid.linestyle': '--'}):

    ax = sns.barplot(data=df_sel, x="sim_cnt", y=f"{metric}", hue="name", hue_order=hue_order)
    #sns.lineplot(data=df, x="ratio", y=f"{metric}", hue="name", hue_order=hue_order, palette=palette)
    #ax = sns.scatterplot(data=df, x="ratio", y=f"{metric}", hue="name", hue_order=hue_order, palette=palette, legend=False)
    ax.set_xticks(x_tick) 
    ax.set_yticks(y_tick) 
    ax.set_xticklabels(x_labels)

#sns.legend(title='Sim. method', fontsize='13')
ax.set_xlabel('Rel. annotation work', fontsize='12')
ax.set_ylabel('Rel. annotation quality', fontsize='12')
plt.legend( loc='lower right', fontsize='12')


plt.tight_layout()
plt.savefig(f"{metric}.png")
plt.show()
