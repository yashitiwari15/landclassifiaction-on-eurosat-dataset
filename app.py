import streamlit as st
from PIL import Image

import numpy as np
import tensorflow as tf

from load import init
from config import idx2class

st.write('''<style>
            body{
            text-align:center;
            }
            </style>''', unsafe_allow_html=True)


st.title('Land Cover Classification')

file_type = 'jpg'
uploaded_file = st.file_uploader("Upload a file", type = file_type)

global model
model = init()
st.text('Loaded ResNet50')

if uploaded_file != None:
    image = Image.open(uploaded_file)
    image = tf.keras.utils.load_img(uploaded_file, target_size=(224,224))
    st.image(image)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr).argmax()
    st.text(f'Predicted Class: {idx2class[prediction]}')
    