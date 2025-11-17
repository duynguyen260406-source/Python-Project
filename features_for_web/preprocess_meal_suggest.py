import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib 

# Load và chuẩn hóa dữ liệu 
df = pd.read_csv("data/RAW_recipes.csv") 

for col in ["tags", "nutrition", "steps", "ingredients"]:
    df[col] = df[col].fillna("[]").apply(literal_eval)

def build_text(row):
    parts = []

    if pd.notna(row["name"]):
        parts.append(str(row["name"]))
    if pd.notna(row["description"]):
        parts.append(str(row["description"]))
    
    parts.append(" ".join(map(str, row["steps"])))
    parts.append(" ".join(map(str, row["tags"])))
    parts.append(" ".join(map(str, row["ingredients"])))

    return " ".join(parts)

df["calories"] = df["nutrition"].str[0]
df["text"] = df.apply(build_text, axis=1)


vectorizer = TfidfVectorizer(
    stop_words="english",      
    ngram_range=(1, 2),        
    min_df=3                   
)

recipe_tfidf = vectorizer.fit_transform(df["text"])

joblib.dump(vectorizer, "artifacts/vectorizer.pkl")
joblib.dump(recipe_tfidf, "artifacts/recipe_tfidf.pkl")
df.to_pickle("artifacts/recipes_df.pkl")