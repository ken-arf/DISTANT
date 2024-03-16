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


fig, (ax1, ax2) = plt.subplots(1,2)


g1 = sns.barplot(x="Iter", y="count", hue="op",\
                data=df1, ax=ax1)

g2 = sns.barplot(x="Iter", y="count", hue="op",\
                data=df2, ax=ax2)


labels = ["chunk", "Iterate"]

#fig.legend([g1, g2], labels=labels, 
#           loc="upper right") 
  
# Adjusting the sub-plots 
plt.subplots_adjust(right=0.9) 
  
plt.show() 




