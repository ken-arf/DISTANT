import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


csv = "./data/ChunkUp_op_count.csv"
#csv = "./data/IterUp_op_count.csv"

df = pd.read_csv(csv)

print(df.columns)

cols = ["Iter", "Match", "Delete", "New", "Update"]
df = df[cols]
print(df)

cols = ["Iter", "Match", "New", "Update"]
sum = df[cols].sum(axis=1).tolist()

df["sum"] = sum

print(df)



df_m = pd.melt(df, 
            id_vars="Iter", 
            var_name="op",
            value_name="count",
        )



print(df_m)

# Plot
fig, ax1 = plt.subplots()
g = sns.barplot(x="Iter", y="count", hue="op",\
                data=df_m, ax=ax1)


plt.show()



