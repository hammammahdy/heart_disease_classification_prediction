import streamlit as st 
import pandas as pd 
import numpy as np 
import pickle
from PIL import Image

def run():

    # membuat title
    st.title('Heart Disease Prediction')

    # membuat subheader
    st.subheader('Data Form Input Heart Disease Classification')

    # menambahkan gambar 
    # image = Image.open('heart_disease_1.jpeg')
    # st.image(image, caption = 'Heart Disease')

    with st.form('Customer Profile'):
    
        # field age
        age = st.number_input('Age', min_value=18, max_value=90, help='Usia')

        # field gender
        gender_options = ["Male", "Female"]
        gender = st.selectbox('Pilih Gender', gender_options)

        # field tenure
        tenure = st.number_input('Tenure', min_value=0, max_value=10, help='How long someone be customer (in years)')

    with st.form('Financial Information'):

        # field pressurehight
        creditscore = st.number_input('Credit Score', min_value=300 , max_value=900, help='How likely to pay a loan back on time, based on information from credit report')

        # field pressurelow
        pressurelow = st.number_input('pressurelow', min_value=60, max_value=250, help='Diastolik (tekanan darah saat jantung beristirahat/diantara detak)')

        # field glucose
        glucose = st.number_input('glucose', min_value=50, max_value=370, help='Kadar gula darah')

        # field kcm
        kcm = st.number_input('kcm', min_value=24, max_value=500, help='Test CK-MB (test pendeteksi enzim kreatin kinase)')

        # field troponin
        troponin = st.number_input('troponin', min_value=1, max_value=500, help='Test troponin (test kadar troponin/protein yang dilepaskan jika terjadi kerusakan jantung)')

        # submit button
        submitted = st.form_submit_button('Predict')


    # inference
    # load all files
    
    with open('scaler.pkl', 'rb') as file_1:
        scaler = pickle.load(file_1)
    with open('model.pkl', 'rb') as file_2:
        model = pickle.load(file_2)

    
    data_inf = {
        'age' : age,
        'gender' : gender,
        'impluse' : impluse,
        'pressurehight' : pressurehight,
        'pressurelow' : pressurelow,
        'glucose' : glucose,
        'kcm' : kcm,
        'troponin' : troponin
    }

    # memasukkan data inference ke dataframe

    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    # logic ketika predict button ditekan
    if submitted:
        data_inf_drop = data_inf.drop(['gender', 'impluse', 'pressurehight', 'pressurelow', 'glucose'], axis=1)

        data_inf_scaled = scaler.transform(data_inf_drop)

    # predict
        y_pred_inf = model.predict(data_inf_scaled)
        st.write('## Heart Disease :', str(int(y_pred_inf)))
        st.write('### Positive : 1, Negative : 2')

if __name__ == '__main__':
    run()

