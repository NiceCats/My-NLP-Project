import streamlit as st
import streamlit.components.v1 as stc

from eda_app import run_eda_dsapp
from ml_app import run_ml_dsapp

html_temp = """
<div style="">
    <h1>Cyberbullying Text Binary Classification</h1>
</div>
"""

desc_temp = """
    Cyberbullying Text Binary Classification

    This app is used to know the step of Cyberbullying Classification

    ### APP Content
    - Exploratory Data Analysis Preprocessing
    - Machine Learning Section
"""

def main():
    st.markdown(html_temp, unsafe_allow_html=True)

    menu = ["Home", "EDA Preprocessing", "ML Section"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.markdown(desc_temp, unsafe_allow_html=True)
    elif choice == "EDA Preprocessing":
        run_eda_dsapp()
    elif choice == "ML Section":
        run_ml_dsapp()

if __name__ == '__main__':
    main()