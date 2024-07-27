import pickle
import streamlit as st
import numpy as np
from skimage.transform import resize
from skimage.io import imread

model=pickle.load(open('model1.pkl','rb'))

st.title('Emotion Recognition')
file=st.file_uploader('Upload an image',type=['png'])

def emotion_rec(spot_bgr):

    flat_data = []
    img_resized = resize(spot_bgr, (30,30,3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)

    y_output = model.predict(flat_data)
    return y_output

if st.button('Predict'):
    image = imread(file)
    result=emotion_rec(image)
    st.image(image)
    if result == 0:
        st.success('angry')
    elif result == 1:
        st.success("Fear ")
    elif result == 2:
        st.success("happy")
    elif result == 3:
        st.success("sad")
    else:
        st.error('No emotion...Wrong Image')
