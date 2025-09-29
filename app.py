from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from mtcnn import MTCNN
from PIL import Image
import streamlit as st
import os
import re


detector = MTCNN()
model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
feature_list=np.array(pickle.load(open('embedding.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

def save_uploaded_image(uploaded_image):
   # Define a reliable uploads folder
   UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "uploads")
   os.makedirs(UPLOADS_DIR, exist_ok=True)

   if uploaded_image is not None:
        # Sanitize filename
        filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in uploaded_image.name)
        save_path = os.path.join(UPLOADS_DIR, filename)

        # Write file
        with open(save_path, "wb") as f:
           f.write(uploaded_image.getbuffer())

        return True





def extract_feature(img_path,model,detector):
    sample_img = cv2.imread(img_path)

    results = detector.detect_faces(sample_img)

    X, Y, width, height = results[0]['box']

    face = sample_img[Y:Y + height, X:X + width]
    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = np.asarray(image)
    face_array = face_array.astype('float32')
    expanded_image = np.expand_dims(face_array, axis=0)
    preprocess_img = preprocess_input(expanded_image)

    results = model.predict(preprocess_img).flatten()
    return  results


def recommend(feature_list,features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

st.title('Which bollywood celebrity are you ?')

uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    if save_uploaded_image(uploaded_image):
        display_image = Image.open(uploaded_image)
        features = extract_feature(os.path.join('uploads',uploaded_image.name),model,detector)
        index_pos=recommend(feature_list,features)
        predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
        col1,col2=st.columns(2)
        with col1:
            st.header('your uploaded image')
            st.image(display_image,width=300)
        with col2:
            st.header("Seems Like " + predicted_actor)
            st.image(filenames[index_pos],width=300)
