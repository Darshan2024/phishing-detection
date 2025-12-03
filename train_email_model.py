# train_email_model.py
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/raw/Phishing_Email.csv"
MODEL_PATH = "email_model.pkl"

def main():
    # 1. Load dataset
    df = pd.read_csv(DATA_PATH)

    # Adjust these column names if your CSV differs
    df = df[["Email Text", "Email Type"]].copy()
    df.rename(columns={"Email Text": "text", "Email Type": "label"}, inplace=True)

    # 2. Basic cleaning
    df.dropna(subset=["text", "label"], inplace=True)
    df.drop_duplicates(subset=["text", "label"], inplace=True)

    # 3. Map labels to 0/1
    label_map = {
        "Safe Email": 0,
        "Phishing Email": 1,
    }
    df = df[df["label"].isin(label_map.keys())].copy()
    df["label"] = df["label"].map(label_map)

    X = df["text"]
    y = df["label"]

    # 4. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Build pipeline: TF-IDF + Logistic Regression
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
        ("model", LogisticRegression(max_iter=1000))
    ])

    # 6. Train
    clf.fit(X_train, y_train)

    # 7. Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # 8. Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    print(f"✅ Saved trained email phishing model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
