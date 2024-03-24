import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_df(csv):

    df = pd.read_csv(csv)

    print(df.columns)

    cols = ["Iter", "Match", "Delete", "New", "Update"]
    df = df[cols]
    print(df)

    cols = ["Iter", "Match", "New", "Update"]
    sum = df[cols].sum(axis=1).tolist()

    df["Gold ann"] = sum

    print(df)

    df_m = pd.melt(df, 
                id_vars="Iter", 
                var_name="op",
                value_name="count",
            )



    print(df_m)
    return df_m


csv = "./data/ChunkUp_op_count.csv"
df1 = get_df(csv)

csv = "./data/IterUp_op_count.csv"
df2 = get_df(csv)


#with sns.set_style("darkgrid", {'grid.linestyle': '--'}):
with sns.axes_style("whitegrid", {'grid.linestyle': '--'}):
    #pal = sns.color_palette("Set2")
    pal = sns.color_palette("tab10")
    sns.set_palette(pal)
    fig, (ax1, ax2) = plt.subplots(1,2)

#sns.color_palette()
#sns.set_style("darkgrid", {'grid.linestyle': '--'})


#print(sns.axes_style())

g1 = sns.barplot(x="Iter", y="count", hue="op",\
                data=df1, ax=ax1)

g2 = sns.barplot(x="Iter", y="count", hue="op",\
                data=df2, ax=ax2)


labels = ["chunk", "Iterate"]

ax1.legend().set_title('')
ax2.legend().set_title('')


plt.setp(ax1.get_legend().get_texts(), fontsize='12') # for legend text
plt.setp(ax2.get_legend().get_texts(), fontsize='12') # for legend text

ax1.set_xlabel('Sim. round (ChunkUpdate)', fontsize='12')
ax2.set_xlabel('Sim. round (IterativeUpdate)', fontsize='12')

ax1.set_ylabel('Count', fontsize='12')
ax2.set_ylabel('Count', fontsize='12')


#fig.legend([g1, g2], labels=labels, 
#           loc="upper right") 
  
# Adjusting the sub-plots 
plt.subplots_adjust(right=0.9) 
plt.tight_layout()
  
plt.show() 




