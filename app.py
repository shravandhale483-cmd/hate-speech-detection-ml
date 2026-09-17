"""
Hate Speech Detection using Machine Learning

Educational NLP project using:
- TF-IDF Vectorization
- Logistic Regression
- NLTK preprocessing
- Gradio interface
"""

import re
import string

import gradio as gr
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# -----------------------------
# 1. NLTK setup
# -----------------------------
nltk.download("stopwords", quiet=True)

STOPWORDS = set(stopwords.words("english"))
STEMMER = SnowballStemmer("english")


# -----------------------------
# 2. Dataset
# -----------------------------
data = {
    "tweet": [
        # Hate speech
        "I hate you",
        "Go die",
        "You are disgusting",
        "You’re worthless",
        "You make me sick",
        "Burn in hell",
        "Kill them all",
        "I despise you",
        "Go die loser",
        "You don’t deserve to live",

        # Offensive language
        "You are bad",
        "You idiot",
        "You are stupid",
        "You’re a fool",
        "You are rude",
        "You talk nonsense",
        "You suck",
        "You dumb person",
        "You are useless",
        "You’re pathetic",

        # Neutral / non-offensive
        "You are amazing",
        "I love you",
        "Have a nice day",
        "What a beautiful day",
        "You did a great job",
        "Let’s help others",
        "I admire your courage",
        "Peace and love to all",
        "You’re so kind",
        "Congratulations on your success",
    ],
    "class": (
        [0] * 10 +  # Hate
        [1] * 10 +  # Offensive
        [2] * 10    # Neutral
    ),
}


df = pd.DataFrame(data)


# -----------------------------
# 3. Text preprocessing
# -----------------------------
def clean_text(text: str) -> str:
    """Clean and stem input text."""
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove words containing digits
    text = re.sub(r"\w*\d\w*", "", text)

    # Tokenize, remove stopwords, and stem
    words = [
        STEMMER.stem(word)
        for word in text.split()
        if word not in STOPWORDS
    ]

    return " ".join(words)


df["clean_tweet"] = df["tweet"].apply(clean_text)

LABELS = {
    0: "🚫 Hate Speech Detected",
    1: "⚠️ Offensive Language Detected",
    2: "✅ No Hate or Offensive Speech",
}

df["label"] = df["class"].map(LABELS)


# -----------------------------
# 4. Train the model
# -----------------------------
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

X = vectorizer.fit_transform(df["clean_tweet"])
y = df["label"].to_numpy()

model = LogisticRegression(max_iter=2000)
model.fit(X, y)

# Training-set accuracy only.
# This should NOT be interpreted as real-world model accuracy.
train_predictions = model.predict(X)
training_accuracy = accuracy_score(y, train_predictions)

print("✅ Model trained successfully!")
print(f"Training accuracy: {training_accuracy:.2%}")
print("⚠️ Note: This is training-set accuracy on a very small custom dataset.")


# -----------------------------
# 5. Prediction function
# -----------------------------
def predict(text: str) -> str:
    """Predict the category of the entered text."""
    if not text or not text.strip():
        return "⚠️ Please enter some text!"

    cleaned = clean_text(text)

    # Handle an input that becomes empty after preprocessing
    if not cleaned:
        return "⚠️ Please enter meaningful text."

    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]

    return prediction


# -----------------------------
# 6. Gradio interface
# -----------------------------
interface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        label="Enter your text",
        placeholder="Type a sentence here...",
        lines=3,
    ),
    outputs=gr.Textbox(label="Prediction"),
    title="🧠 Hate Speech Detection using Machine Learning",
    description=(
        "Classifies text as Hate Speech, Offensive Language, "
        "or Neutral Speech using TF-IDF and Logistic Regression."
    ),
    examples=[
        ["You are amazing"],
        ["You are stupid"],
        ["I hate you"],
    ],
)


# -----------------------------
# 7. Run application
# -----------------------------
if __name__ == "__main__":
    interface.launch()
