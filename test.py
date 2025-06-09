import pandas as pd

df = pd.read_csv("final_balanced_dataset.csv")

# Count how many Hits and Misses per experience level
print(df.groupby(['shooter_experience', 'target_hit']).size().unstack())
