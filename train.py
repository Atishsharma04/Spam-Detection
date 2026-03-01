import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only needed columns
data = data[['v1','v2']]

# Rename columns
data.columns = ['label','message']


print(data.head())
print(data.info())

# Convert labels to numbers
data['label'] = data['label'].map({
    'ham':0,
    'spam':1
})

print(data.head())


# Convert text into numbers

vectorizer = TfidfVectorizer()

# vectorizer = CountVectorizer()

X = vectorizer.fit_transform(data['message'])

y = data['label']

print(X.shape)

###
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

print(X_train.shape)

# Create model
model = MultinomialNB()

# Train model
model.fit(X_train, y_train)

print("Model trained successfully!")

# Predict
predictions = model.predict(X_test)

print(predictions[:10])

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)


# Step 12 - Manual prediction

message = input("Enter a message: ")

# Convert text into numbers
message_vector = vectorizer.transform([message])

# Predict
result = model.predict(message_vector)

if result[0] == 1:
    print("Spam Message")
else:
    print("Not Spam Message")