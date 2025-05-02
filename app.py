import streamlit as st
import joblib
import re
import pandas as pd
import altair as alt

svm_model = joblib.load("hate_speech_svm_model.pkl")
vectorizer = joblib.load("/Users/sankar/hatespeech detection/tfidf_vectorizer.pkl")

def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    text = text.lower()
    return text

st.set_page_config(page_title="Hate Speech Detection", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    body {
        font-family: 'Roboto', sans-serif;
    }

    .main {
        animation: fadeIn 2s;
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
    }

    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        animation: pulse 2s infinite;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s;
    }

    .stButton>button:hover {
        transform: scale(1.05);
    }

    .stTextArea textarea {
        height: 150px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: black;
        text-align: center;
        padding: 10px;
        box-shadow: 0 -2px 4px rgba(0, 0, 0, 0.1);
    }

    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }

    @keyframes pulse {
        0% {transform: scale(1);}
        50% {transform: scale(1.05);}
        100% {transform: scale(1);}
    }

    @media (max-width: 768px) {
        .main {
            padding: 10px;
        }
        .stButton>button {
            width: 100%;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align:center; animation: fadeIn 3s;'>🛡️ Hate Speech Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; animation: fadeIn 4s;'>This app detects whether a given sentence is <strong>Hate Speech</strong>, <strong>Offensive</strong>, or <strong>Neither</strong>.</p>", unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Single Detection", "Batch Detection", "About"])

if page == "Single Detection":
    st.header("Single Sentence Detection")
    user_input = st.text_area("✍️ Enter a sentence for detection", "")

    if st.button("Detect"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            cleaned_input = clean_text(user_input)
            try:
                input_tfidf = vectorizer.transform([cleaned_input])
                prediction = svm_model.predict(input_tfidf)[0]

                label_map = {
                    0: "Hate Speech",
                    1: "Neither",
                    2: "Offensive"
                }
                st.success(f"🧠 **Prediction**: {label_map.get(prediction, 'Unknown')}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

elif page == "Batch Detection":
    st.header("Batch Detection")
    uploaded_file = st.file_uploader("Upload a CSV file with sentences", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'sentence' in df.columns:
            df['cleaned_sentence'] = df['sentence'].apply(clean_text)
            try:
                input_tfidf = vectorizer.transform(df['cleaned_sentence'])
                predictions = svm_model.predict(input_tfidf)
                df['prediction'] = predictions

                label_map = {
                    0: "Hate Speech",
                    1: "Neither",
                    2: "Offensive"
                }
                df['prediction'] = df['prediction'].map(label_map)

                st.success("Predictions completed!")
                st.dataframe(df[['sentence', 'prediction']])

                st.subheader("Prediction Distribution")
                chart = alt.Chart(df).mark_bar().encode(
                    x='prediction:N',
                    y='count():Q',
                    color='prediction:N'
                ).properties(
                    width=600,
                    height=400
                )
                st.altair_chart(chart, use_container_width=True)

            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.warning("The CSV file must contain a column named 'sentence'.")

elif page == "About":
    st.header("About")
    st.write("""
    This application uses a Support Vector Machine (SVM) model to detect hate speech, offensive language, or neither in a given sentence.
    The model has been trained on a dataset of labeled sentences and uses TF-IDF vectorization to transform text data into numerical features.
    """)
    st.write("**Contact Information:**")
    st.write("**Name:** Sankara Narayanan S")
    st.write("**Email:** [sankarssn0711@gmail.com](mailto:sankarssn0711@gmail.com)")
    st.write("**Phone:** +91 7010577232")

st.markdown('<div class="footer">© 2023 Hate Speech Detection App</div>', unsafe_allow_html=True)
