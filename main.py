import cv2
import streamlit as st
from deepface import DeepFace
from skimage.io import imread

st.title('Emotion Recognition')
n=st.text_input("Upload Picture:",placeholder="Camera or files")
if(n.lower()=='camera'):
    file=st.camera_input('Click Your Picture')
elif(n.lower()=='files'):
    file=st.file_uploader('Choose a File')
else:
    st.error('TRY AGAIN')

if st.button('Predict'):
    img=imread(file)
    pred=DeepFace.analyze(img)
    st.title("Image")
    st.image(img)
    st.title("Emotion :")
    st.success(pred[0]['dominant_emotion'])
    st.title("Gender :")
    st.success(pred[0]['dominant_gender'])
    st.title("Age :")
    st.success(pred[0]['age'])