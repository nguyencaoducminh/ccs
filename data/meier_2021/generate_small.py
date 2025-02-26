import numpy as np
import pandas as pd

random_seed = 2048

df = pd.read_csv('SourceData_Figure_1.csv')

# df.info()
# exit(0)

random = input("Do you want to randomize the data? (y/n): ")
if random.lower() == 'y':
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

n_data = int(input("Number of data points: "))

if n_data >= len(df):
    n_data = len(df)

df = df[:n_data]

df.to_csv('./meier_small_new.csv', index=False)

print("Data saved to meier_small_new.csv. Please rename to meier_small.csv and use --dataset small")