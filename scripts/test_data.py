import pandas as pd
import os

base_path = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(base_path, "data", "movies.csv")

df = pd.read_csv(file_path)

# Rename columns
df = df.rename(columns={
    'Series_Title': 'movie',
    'Overview': 'story'
})

# Keep only needed columns
df = df[['movie', 'story']]

# Remove null
df = df.dropna()

print(df.head())

# Save cleaned file
clean_path = os.path.join(base_path, "data", "clean_movies.csv")
df.to_csv(clean_path, index=False)

print("✅ Clean dataset saved")