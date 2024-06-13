import streamlit as st
import numpy as np
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import pickle

# Misalkan model dan tfidf_vectorizer sudah dilatih dan disimpan dalam file .pkl
# Load model dan vectorizer
model = pickle.load(open("path_to_your_trained_model.pkl", "rb"))
tfidf_vectorizer = pickle.load(open("path_to_your_tfidf_vectorizer.pkl", "rb"))

def predict_sentiment(text):
    # Transformasi teks menggunakan TF-IDF Vectorizer
    transformed_text = tfidf_vectorizer.transform([text])
    # Prediksi menggunakan model
    prediction = model.predict(transformed_text)
    return prediction[0]

# Membuat aplikasi Streamlit
st.title("Aplikasi Prediksi Sentiment Cyberbullying")

# Input teks dari pengguna
user_input = st.text_area("Masukkan teks tweet Anda di sini:")

if st.button("Prediksi"):
    # Prediksi sentiment
    result = predict_sentiment(user_input)
    # Menampilkan hasil
    if result == 0:
        st.write("Sentiment: Religion")
    elif result == 1:
        st.write("Sentiment: Age")
    elif result == 2:
        st.write("Sentiment: Ethnicity")
    elif result == 3:
        st.write("Sentiment: Gender")
    else:
        st.write("Sentiment: Not Cyberbullying")