import joblib
import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_pickle("artifacts/recipes_df.pkl")
vectorizer = joblib.load("artifacts/vectorizer.pkl")
recipe_tfidf = joblib.load("artifacts/recipe_tfidf.pkl")

def extract_calories(user_query: str):
    text = user_query.lower()
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:calories|calorie|cal|cals|calo|kcalories|kcalorie|kcalo|kcals|kcal)', text)
    if m:
        return float(m.group(1))
    return None

def recommend_recipes(user_query, top_k=5, alpha = 0.5):
    query_vec = vectorizer.transform([user_query])  
    sims_text = cosine_similarity(query_vec, recipe_tfidf)[0]  

    target_cal = extract_calories(user_query)

    if target_cal is not None:
        cal_values = df["calories"].values

        diff = np.abs(cal_values - target_cal)
        cal_sim = 1/(1+diff)

        text_norm = (sims_text - sims_text.min()) / (sims_text.max() - sims_text.min() + 1e-9)
        cal_norm  = (cal_sim  - cal_sim.min())  / (cal_sim.max()  - cal_sim.min()  + 1e-9)

        sims = alpha * text_norm + (1 - alpha)*cal_norm
    else:
        sims = sims_text

    top_idx = np.argsort(-sims)[:top_k]
 
    results = df.loc[top_idx, ["name", "calories", "steps", "ingredients"]].copy()

    return results

if __name__ == "__main__":
    user = "I want a 1000 calories meal with chicken and corn"
    result = recommend_recipes(user)
    result.to_csv("test.csv", index = False)