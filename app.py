
from fastai.vision import *
import numpy as np
import streamlit as st
import PIL
from streamlit import caching
learn = load_learner('')

import time
import cv2    
def main():
  try:
    st.markdown("<h1 style='text-align: center; color: red;'>App Name</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'>Common Objects Detection</h2>", unsafe_allow_html=True)
    st.title("Live Webcam")
    run = st.checkbox('Click to Start')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    def func(x):
        if str(x)=='basket_bin':
            widget.write("<h2 style='text-align: center; color: red;'>Basket Bin</h2>", unsafe_allow_html=True)
        elif str(x)=='bed':
            widget.write("<h2 style='text-align: center; color: red;'>Bed</h2>", unsafe_allow_html=True)
        elif str(x)=='bench':
            widget.write("<h2 style='text-align: center; color: red;'>Bench</h2>", unsafe_allow_html=True)
        elif str(x)=='cabinet':
            widget.write("<h2 style='text-align: center; color: red;'>Cabinet</h2>", unsafe_allow_html=True)
        elif str(x)=='call_bell':
            widget.write("<h2 style='text-align: center; color: red;'>Call Bell</h2>", unsafe_allow_html=True)
        elif str(x)=='cane_stick':
            widget.write("<h2 style='text-align: center; color: red;'>Cane Stick</h2>", unsafe_allow_html=True)
        elif str(x)=='chair':
            widget.write("<h2 style='text-align: center; color: red;'>Chair</h2>", unsafe_allow_html=True)
        elif str(x)=='door':
            widget.write("<h2 style='text-align: center; color: red;'>Door</h2>", unsafe_allow_html=True)
        elif str(x)=='electric_socket':
            widget.write("<h2 style='text-align: center; color: red;'>Electric Socket</h2>", unsafe_allow_html=True)
        elif str(x)=='fan':
            widget.write("<h2 style='text-align: center; color: red;'>Fan</h2>", unsafe_allow_html=True)
        elif str(x)=='fire_extinguisher':
            widget.write("<h2 style='text-align: center; color: red;'>Fire Extinguisher</h2>", unsafe_allow_html=True)
        elif str(x)=='handrail':
            widget.write("<h2 style='text-align: center; color: red;'>Handrail</h2>", unsafe_allow_html=True)
        elif str(x)=='human_being':
            widget.write("<h2 style='text-align: center; color: red;'>Human Being</h2>", unsafe_allow_html=True)
        elif str(x)=='rack':
            widget.write("<h2 style='text-align: center; color: red;'>Rack</h2>", unsafe_allow_html=True)
        elif str(x)=='refrigerator':
            widget.write("<h2 style='text-align: center; color: red;'>Refrigerator</h2>", unsafe_allow_html=True)
        elif str(x)=='shower':
            widget.write("<h2 style='text-align: center; color: red;'>Shower</h2>", unsafe_allow_html=True)
        elif str(x)=='sink':
            widget.write("<h2 style='text-align: center; color: red;'>Sink</h2>", unsafe_allow_html=True)
        elif str(x)=='sofa':
            widget.write("<h2 style='text-align: center; color: red;'>Sofa</h2>", unsafe_allow_html=True)
        elif str(x)=='table':
            widget.write("<h2 style='text-align: center; color: red;'>Table</h2>", unsafe_allow_html=True)
        elif str(x)=='television':
            widget.write("<h2 style='text-align: center; color: red;'>Television</h2>", unsafe_allow_html=True)
        elif str(x)=='toilet_seat':
            widget.write("<h2 style='text-align: center; color: red;'>Toilet Seat</h2>", unsafe_allow_html=True)
        elif str(x)=='walker':
            widget.write("<h2 style='text-align: center; color: red;'>Walker</h2>", unsafe_allow_html=True)
        elif str(x)=='wardrobe':
            widget.write("<h2 style='text-align: center; color: red;'>Wardrobe</h2>", unsafe_allow_html=True)
        elif str(x)=='water_dispencer':
            widget.write("<h2 style='text-align: center; color: red;'>Water Dispencer</h2>", unsafe_allow_html=True)
        elif str(x)=='wheelchair':
            widget.write("<h2 style='text-align: center; color: red;'>Wheelchair</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: red;'>Classifying...</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: black;'>The Object is a :</h3>", unsafe_allow_html=True)
    temp=''
    widget = st.empty()
    while run:
        caching.clear_cache()
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)  
        img_t = pil2tensor(frame, np.float32)
        img_t.div_(255.0)
        image = Image(img_t)
        a,cat_tensor,c = learn.predict(image)
        if temp!=str(a):
            func(a)
        temp=str(a)
    else:
        st.write('Check the box to Start Running')
  except Exception:
    pass
if __name__ == '__main__':
  main()
