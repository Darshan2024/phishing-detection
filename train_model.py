import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# This is the usual Kaggle SMS spam format: v1 = label, v2 = text
df = df[["v1", "v2"]]
df.columns = ["label", "text"]

# Map labels to 0/1
X = df["text"]
y = df["label"].map({"ham": 0, "spam": 1})

# 2. Split train/test (just to see that it works)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Build pipeline: TF-IDF (with English stopwords) + Naive Bayes
clf = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("model", MultinomialNB())
])

# 4. Train
clf.fit(X_train, y_train)

# 5. Quick evaluation
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 6. Save the whole pipeline as model.pkl
with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("✅ Training complete. Saved model.pkl")
