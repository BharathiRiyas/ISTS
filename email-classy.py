# ==========================================
# Spam Email Classification using NLP & ML
# ==========================================

import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------------------
# Download NLTK Resources
# ------------------------------------------

nltk.download('punkt')
nltk.download('stopwords')

# ------------------------------------------
# Sample Dataset
# ------------------------------------------

data = {
    "label": [
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham"
    ],

    "message": [
        "Congratulations! You have won 10000 rupees.",
        "Meeting is scheduled at 10 AM tomorrow.",
        "Claim your free gift now.",
        "Project submission deadline is tomorrow.",
        "Win a brand new iPhone today.",
        "Please attend the training session.",
        "Limited offer! Click here to win cash prize.",
        "Can we discuss the project today?",
        "You are selected for a lucky draw.",
        "Lunch meeting at 1 PM."
    ]
}

df = pd.DataFrame(data)

print("\nOriginal Dataset")
print(df)

# ------------------------------------------
# Text Preprocessing
# ------------------------------------------

stop_words = set(stopwords.words('english'))

def preprocess(text):

    # Convert to lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Stopword Removal
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Join back into sentence
    return " ".join(tokens)

# Apply preprocessing

df["clean_message"] = df["message"].apply(preprocess)

print("\nAfter Preprocessing")
print(df[["message", "clean_message"]])

# ------------------------------------------
# Convert Labels to Numbers
# ham = 0
# spam = 1
# ------------------------------------------

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# ------------------------------------------
# TF-IDF Vectorization
# ------------------------------------------

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["clean_message"])

y = df["label"]

print("\nTF-IDF Shape:")
print(X.shape)

# ------------------------------------------
# Train-Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------------------------------
# Train Model
# ------------------------------------------

model = MultinomialNB()

model.fit(X_train, y_train)

# ------------------------------------------
# Prediction
# ------------------------------------------

y_pred = model.predict(X_test)

# ------------------------------------------
# Accuracy
# ------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(round(accuracy * 100, 2), "%")

# ------------------------------------------
# Classification Report
# ------------------------------------------

print("\nClassification Report")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# ------------------------------------------
# Test New Emails
# ------------------------------------------

print("\n==============================")
print("Spam Email Prediction")
print("==============================")

while True:

    email = input("\nEnter Email Text: ")

    clean_email = preprocess(email)

    email_vector = vectorizer.transform(
        [clean_email]
    )

    prediction = model.predict(
        email_vector
    )

    if prediction[0] == 1:
        print("Result: SPAM EMAIL")
    else:
        print("Result: NOT SPAM")

    choice = input(
        "\nCheck another email? (yes/no): "
    )

    if choice.lower() != "yes":
        break

print("\nProgram Finished")