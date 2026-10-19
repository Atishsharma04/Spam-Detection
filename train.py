import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

# Keep required columns
data = data[["v1", "v2"]].copy()
data.columns = ["label", "message"]

# Convert labels to numbers
data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

# Remove missing or invalid rows
data = data.dropna(subset=["label", "message"])
data["label"] = data["label"].astype(int)

X = data["message"]
y = data["label"]

# Split data BEFORE fitting the vectorizer
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert text into numerical features
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

print("Model trained successfully!")

# Evaluate model
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(
    y_test,
    predictions,
    target_names=["Ham", "Spam"]
))

# Manual prediction
while True:
    message = input("\nEnter a message (or type 'exit' to quit): ")

    if message.lower() == "exit":
        break

    message_vector = vectorizer.transform([message])
    result = model.predict(message_vector)

    if result[0] == 1:
        print("Spam Message")
    else:
        print("Not Spam Message")
