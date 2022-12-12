import cv2
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from gen_db import img_to_vec
from sklearn.neighbors import NearestNeighbors


@st.cache(allow_output_mutation=True)
def load_db():
    df = pd.read_pickle('db.pickle')
    nbrs = NearestNeighbors(n_neighbors=5, metric='cosine', n_jobs=-1)
    nbrs.fit(np.stack(df['vec'].to_numpy()))
    return df, nbrs


@st.cache(allow_output_mutation=True)
def load_model():
    with open('model.pickle', 'rb') as f:
        return pickle.load(f)


db, neighbours = load_db()
model, n = load_model()
st.title('Image Search')
uploaded_file = st.file_uploader('Choose an image...', type=['png', 'jpg', 'jpeg', 'webp', 'tiff'])
if uploaded_file is not None:
    file_arr = np.frombuffer(uploaded_file.getbuffer(), dtype='uint8')
    img = cv2.imdecode(file_arr, cv2.IMREAD_GRAYSCALE)

    vec = img_to_vec(img, model, n)
    indices = neighbours.kneighbors(vec.reshape(1, -1), return_distance=False)[0]
    paths = np.hstack(db.loc[indices, ['path']].values)

    for path in paths:
        st.image(path, caption=path)
