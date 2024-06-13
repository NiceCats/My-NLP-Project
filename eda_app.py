import streamlit as st
import numpy as np
import joblib
import os
import pickle

Dataset_Info =  """
                    Dataset Cyberbullying Tweets
                    Dataset ini berisi lebih dari 47.000 tweet Data telah diseimbangkan untuk memuat sekitar 8.000 instansi untuk setiap kelas.
                    Informasi Kategori:
                    - Usia: Rentang usia dari pengguna tweet.
                    - Etnis: Informasi tentang latar belakang etnis dari pengguna.
                    - Jenis Kelamin: Jenis kelamin dari pengguna tweet.
                    - Agama: Informasi tentang agama pengguna tweet.
                    
                    Jenis Penindasan Maya Lainnya: Kategori-kategori khusus dari penindasan maya yang mungkin terjadi.
                    Bukan Penindasan Maya: Kategori untuk tweet yang tidak terkait dengan penindasan maya.
                    Source : 
                    https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification
                """

# Dictionary for images before cleaning
images_before = {
    "Cyberbullying Type Count": "images/Output per cyberbullying type.png",
    "Average Tweet Length": "images/average tweet length.png",
    "Word Cloud for not cyberbullying": "images/word cloud for non cyberbullying before.png",
    "Word Cloud for gender": "images/word cloud for gender before.png",
    "Word Cloud for religion": "images/word cloud for religion before.png",
    "Word Cloud for age": "images/word cloud for age before.png",
    "Word Cloud for ethnicity": "images/word cloud for ethnicity before.png",
    "Word Cloud for other cyberbullying": "images/word cloud for other cyberbullying before.png"
}

image_cleaning = {
    "Process for cleaning data": "images/cleaning data.png",
    "Results from cleaned data": "images/result from cleaning data.png",
    "Eesults of the data table after data cleaning": "images/df cleaning data head.png"
}

# Dictionary for images after cleaning
images_after = {
    "Average Tweet Length": "images/average tweet length after.png",
    "Word Cloud for not cyberbullying": "images/word cloud for non cyberbullying after.png",
    "Word Cloud for gender": "images/word cloud for gender after.png",
    "Word Cloud for religion": "images/word cloud for religion after.png",
    "Word Cloud for age": "images/word cloud for age after.png",
    "Word Cloud for ethnicity": "images/word cloud for ethnicity after.png"
}

@st.cache_data
def load_model(model_file):
    model_path = os.path.join("models", model_file)
    loaded_model = joblib.load(open(model_path, "rb"))
    return loaded_model

def run_eda_dsapp():
    st.subheader("EDA section")

    with st.expander("Dataset Information"):
        st.markdown(Dataset_Info)
        
    st.subheader("EDA preprocessing before cleaning")
    # Loop through images_before dictionary
    for caption, image_path in images_before.items():
        with st.expander(caption):
            st.image(image_path, caption=caption)

    st.subheader("Process data cleaning")
    # Loop through image_cleaning dictionary
    for caption, image_path in image_cleaning.items():
        with st.expander(caption):
            st.image(image_path, caption=caption)

    st.subheader("EDA preprocessing after cleaning")
    # Loop through images_after dictionary
    for caption, image_path in images_after.items():
        with st.expander(caption):
            st.image(image_path, caption=caption)