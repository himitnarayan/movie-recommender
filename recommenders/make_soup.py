import pandas as pd
from ast import literal_eval
from tqdm import tqdm

tqdm.pandas()

print("Loading cleaned data...")
df = pd.read_csv("clean_movies.csv")

# ---------- Helper functions ----------

def parse_names(text):
    """
    Convert JSON string like:
    '[{"id": 28, "name": "Action"}, ...]'
    into: ['action', 'adventure']
    """
    try:
        items = literal_eval(text)
        return [i['name'].replace(" ", "").lower() for i in items]
    except:
        return []

def clean_text(text):
    if isinstance(text, str):
        return text.lower()
    return ""

# ---------- Parse genres & keywords ----------

print("Parsing genres and keywords...")

df['genres'] = df['genres'].progress_apply(parse_names)
df['keywords'] = df['keywords'].progress_apply(parse_names)

# Clean overview
df['overview'] = df['overview'].progress_apply(clean_text)

# ---------- Create metadata soup ----------

print("Creating metadata soup...")

def create_soup(row):
    return (
        " ".join(row['genres']) + " " +
        " ".join(row['keywords']) + " " +
        row['overview']
    )

df['soup'] = df.progress_apply(create_soup, axis=1)

# Keep only needed columns
final_df = df[['id', 'title', 'soup', 'poster_path', 'popularity']]

print("Final shape:", final_df.shape)

# Save for next step
final_df.to_csv("movies_soup.csv", index=False)

print("✅ Step 2 complete. Saved movies_soup.csv")
