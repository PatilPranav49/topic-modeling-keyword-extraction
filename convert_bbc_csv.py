import pandas as pd
import os

# load csv
df = pd.read_csv("train.csv")

# create raw folder
os.makedirs("data/raw", exist_ok=True)

# clear old data
for file in os.listdir("data/raw"):
    os.remove(os.path.join("data/raw", file))

# IMPORTANT: check column names
print(df.columns)

# usually BBC dataset has columns like: category, text
# adjust if needed

for i, row in df.iterrows():
    text = str(row["text"])   # change if column name different
    
    with open(f"data/raw/doc_{i}.txt", "w", encoding="utf-8") as f:
        f.write(text)

print("✅ Conversion done")