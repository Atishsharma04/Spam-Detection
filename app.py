import streamlit as st
import pickle

# Page settings
st.set_page_config(
    page_title="SpamShield AI | Spam Detection",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #172554);
    color: #f8fafc;
}

.main .block-container {
    max-width: 950px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    text-align: center;
    padding: 25px 10px;
}

.hero h1 {
    font-size: 46px;
    color: #60a5fa;
    font-weight: 800;
    margin-bottom: 10px;
}

.hero p {
    font-size: 18px;
    color: #e2e8f0;
}

.card {
    background: #1e293b;
    border: 1px solid #475569;
    padding: 25px;
    border-radius: 18px;
    margin: 20px 0;
}

.stTextArea textarea {
    background-color: #ffffff !important;
    color: #111827 !important;
    font-size: 18px !important;
    line-height: 1.6 !important;
    border: 2px solid #60a5fa !important;
    border-radius: 12px !important;
    min-height: 240px !important;
    padding: 16px !important;
    caret-color: #111827 !important;
}

.stTextArea textarea::placeholder {
    color: #6b7280 !important;
    opacity: 1 !important;
}

.stTextArea label {
    color: #f8fafc !important;
    font-size: 18px !important;
    font-weight: 600 !important;
}

div.stButton > button {
    width: 100%;
    background-color: #2563eb;
    color: #ffffff !important;
    border: 1px solid #3b82f6;
    border-radius: 10px;
    min-height: 48px;
    font-size: 16px;
    font-weight: 700;
    transition: 0.2s;
}

div.stButton > button:hover {
    background-color: #1d4ed8;
    border-color: #93c5fd;
    color: #ffffff !important;
}

.stApp p,
.stApp label,
.stApp .stMarkdown {
    color: #f1f5f9;
}

.footer {
    text-align: center;
    color: #cbd5e1;
    padding: 25px 0;
    font-size: 14px;
    line-height: 1.8;
}

hr {
    border-color: #334155;
}
</style>
""", unsafe_allow_html=True)


# Load the saved model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    return model, vectorizer


try:
    model, vectorizer = load_model()
except Exception:
    st.error(
        "Unable to load the model. Check that model.pkl "
        "and vectorizer.pkl exist in your project folder."
    )
    st.stop()


# Display the heading
st.markdown("""
<div class="hero">
    <h1>🛡️ SpamShield AI</h1>
    <p>
        Intelligent message classification powered by Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()


# Display the message section
st.markdown("""
<div class="card">
    <h2>📩 Analyse Your Message</h2>
    <p>
        Check whether a message is spam or a legitimate message.
    </p>
</div>
""", unsafe_allow_html=True)


# Sample message buttons
st.markdown("### 🧪 Try a sample message")

col1, col2 = st.columns(2)

with col1:
    if st.button("🎁 Load Sample Spam", use_container_width=True):
        st.session_state["message_input"] = (
            "Congratulations! You have won a free iPhone. "
            "Click here to claim your prize now!"
        )

with col2:
    if st.button("💬 Load Sample Normal", use_container_width=True):
        st.session_state["message_input"] = (
            "Hey, are we meeting tomorrow at 10 AM?"
        )


# Message input
message = st.text_area(
    "Enter your message",
    placeholder="Type or paste your message here...",
    height=240,
    max_chars=10000,
    key="message_input"
)


# Predict the message category
if st.button("🔍 Analyse Message", use_container_width=True):

    if not message.strip():
        st.warning("Please enter a message before analysing.")

    else:
        try:
            message_vector = vectorizer.transform([message])
            prediction = model.predict(message_vector)[0]

            st.divider()
            st.markdown("### 📊 Analysis Result")

            if prediction == 1:
                st.error("🚨 SPAM DETECTED")
                st.markdown("""
                This message has been classified as **spam** by the model.

                Be cautious with suspicious links, unexpected prizes,
                and requests for personal information.
                """)

            else:
                st.success("✅ MESSAGE APPEARS LEGITIMATE")
                st.markdown("""
                This message has been classified as **ham (not spam)**
                by the model.
                """)

            # Show prediction confidence
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(message_vector)[0]
                confidence = float(probabilities.max()) * 100

                st.markdown("#### Model prediction confidence")
                st.progress(int(confidence))
                st.write(f"Model confidence: **{confidence:.2f}%**")

                st.caption(
                    "This is the model's predicted probability, "
                    "not a guarantee that the prediction is correct."
                )

            st.info(
                "AI predictions can make mistakes. "
                "Always verify suspicious messages independently."
            )

        except Exception as error:
            st.error("An error occurred while analysing the message.")
            st.exception(error)


# Display project information
st.divider()

with st.expander("ℹ️ About SpamShield AI"):
    st.write("""
    **SpamShield AI** classifies text messages as spam or ham.

    **Technologies used:**
    - Python
    - Pandas
    - Scikit-learn
    - TF-IDF text feature extraction
    - Multinomial Naive Bayes
    - Streamlit
    """)


# Display the footer
st.markdown("""
<div class="footer">
    <strong>SpamShield AI</strong><br>
    Built with Python · Scikit-learn · Streamlit<br>
    Machine Learning Project
</div>
""", unsafe_allow_html=True)
