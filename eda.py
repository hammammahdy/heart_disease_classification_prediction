# import libraries 
import streamlit as st 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import plotly.express as px 
from PIL import Image

def run():

    # membuat title
    st.title('Heart Disease Prediction')

    # membuat subheader
    st.subheader('EDA untuk analisa dataset heart disease classification')

    # menambahkan gambar 
    image = Image.open('heart_disease_1.jpeg')
    st.image(image, caption = 'Heart Disease')

    # menambahkan deskripsi
    st.write('Web App ini berfungsi untuk memprediksi penyakit jantung menggunakan machine learning (by: Hammam Mahdy)')

    # membuat markdown
    st.markdown('---------------')

    # dataframe
    df = pd.read_csv('Heart Attack.csv')

    # membuat bar plot
    st.write('#### Plot Gender')
    fig = plt.figure(figsize=(15,5))
    sns.countplot(x=df['gender'], data=df)

    # membuat histogram plot
    st.write('#### Histogram Age')
    fig = plt.figure(figsize=(15,5))
    sns.histplot(df['age'], bins=30, kde=True)
    st.pyplot(fig)

    # membuat bar plot berdasarkan inputan user
    st.write('#### Bar Plot berdasarkan pilihan user')
    option = st.selectbox('pilih column :', ('gender', 'class'))
    fig = plt.figure(figsize=(15,5))
    sns.countplot(x=option, data=df)
    st.pyplot(fig)

    # membuat histogram berdasarkan inputan user
    st.write('#### Histogram Plot berdasarkan pilihan user')
    option = st.selectbox('pilih column :', ('age', 'impluse', 'pressurehight', 'pressurelow', 'glucose', 'kcm', 'troponin'))
    fig = plt.figure(figsize=(15,5))
    sns.histplot(df[option], bins=30, kde=True)
    st.pyplot(fig)
    
if __name__ == '__main__':
    run()