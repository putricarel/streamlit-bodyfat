import pickle
import streamlit as st

model = pickle.load(open('Estimasi_BodyFat.sav', 'rb'))

st.title('Estimasi BodyFat Laki-laki')

Density = st.number_input('Input Kepadatan ditentukan dari penimbangan di bawah air')
Age = st.number_input('Input Usia (tahun)')
Weight = st.number_input('Input Berat (pon)')
Height = st.number_input('Input Tinggi (inchi)')
Neck = st.number_input('Input Lingkar leher (cm)')
Chest = st.number_input('Input Lingkar dada (cm)')
Abdomen = st.number_input('Input Lingkar perut 2 (cm)')
Hip = st.number_input('Input Lingkar pinggul (cm)')
Thigh = st.number_input('Input Lingkar paha (cm)')

predict = ''

if st.button('Estimasi bodyfat'):
    predict = model.predict(
        [[Density, Age, Weight, Height, Neck, Chest, Abdomen, Hip, Thigh]]
    )
    st.write ('Estimasi bodyfat dalam persen : ', predict)