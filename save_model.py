import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')

# Keep useful columns
data = data[['v1','v2']]
data.columns = ['label','message']

# Convert labels to numbers
data['label'] = data['label'].map({
    'ham':0,
    'spam':1
})

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['message'])
y = data['label']

# Train model
model = MultinomialNB()
model.fit(X,y)

# Save model and vectorizer
pickle.dump(model, open("model.pkl","wb"))
pickle.dump(vectorizer, open("vectorizer.pkl","wb"))

print("Model saved successfully!")

