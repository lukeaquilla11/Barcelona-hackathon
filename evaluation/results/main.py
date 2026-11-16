import pandas as pd
df = pd.read_csv("Nov_11_final.csv")
print(df[11185:18686]["label"].value_counts())