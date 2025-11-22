import gdown
import os

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

ARTIFACT_FILES = {
    "best_calorie_model.pkl": "https://drive.google.com/uc?id=1yG4WkEUPlCbCj2PVOCUB8bNQ74qkek2E",
    "recipes_df.pkl": "https://drive.google.com/uc?id=1jDcArGVDF4yQLb4TEynV4Y2r6dTberKU",
    "recipe_tfidf.pkl": "https://drive.google.com/uc?id=1IKf9HpeeNYiuTbVCifr5QQLPQKaFrRCX",
    "vectorizer.pkl": "https://drive.google.com/uc?id=1NQTytH8N3vKVYIwAoFgA_D5nguT39jfa",
}

DATA_FILES = {
    "food_calories_vi.csv": "https://drive.google.com/uc?id=19Cs9GrRDRaLliacL1CFUiR52135D7ljn",
    "exercise_dataset (1).csv": "https://drive.google.com/uc?id=12wivBtIZl3L0D-2bFn10BCF9SoMnIHEx",
}

RAW_FILES = {
        "train.csv": "https://drive.google.com/uc?id=1QCkHkERxJQ7kwd_TbwWBrLPXAXVDH6fF",
        "RAW_recipes.csv": "https://drive.google.com/uc?id=1NWb6zX6Pz52o_yBaZST2VvuiiW6gsRRa"
}

def ensure_files():
    for fname, url in ARTIFACT_FILES.items():
        path = os.path.join(ARTIFACT_DIR, fname)
        if not os.path.exists(path):
            gdown.download(url, path, quiet=False)

    for fname, url in DATA_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            gdown.download(url, path, quiet=False)

if __name__ == "__main__":
    ensure_files()
    for fname, url in RAW_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            gdown.download(url, path, quiet=False)