import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl","rb"))

# Load vectorizer
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

st.title("Spam Detection App")

message = st.text_area("Enter Message")

if st.button("Check Spam"):

    message_vector = vectorizer.transform([message])

    result = model.predict(message_vector)

    if result[0] == 1:
        st.error("Spam Message")
    else:
        st.success("Not Spam Message")