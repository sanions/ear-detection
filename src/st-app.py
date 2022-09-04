import imghdr
import streamlit as st
import cv2
import numpy as np
from PIL import Image

from main import main

st.title("Ear Detection")

st.write("Refer to the Github repo at https://github.com/sanions/whisper-ear-detection for a description, instructions, and example inputs.")

uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_arr = np.array(image)
    st.header("Uploaded Image")
    st.image(img_arr, width=500)
    img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    d1, d2 = main(input_img=img)

    st.header("Reference Object")
    st.image('./images/progress/reference_object.jpg', width=500)

    st.header("Identify Ear")
    st.image('./images/progress/detected_ears.jpg', width=500)

    st.header("Landmarks")
    st.image('./images/result/img_landmarks.jpg', width=500)

    st.write(f"The distance is between {d1} and {d2} inches.")
