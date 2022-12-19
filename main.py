import torch
import open_clip
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from sklearn.neighbors import NearestNeighbors


@st.cache(allow_output_mutation=True)
def load_db():
    df = pd.read_pickle('db.pickle')
    nbrs = NearestNeighbors(n_neighbors=5, metric='cosine', n_jobs=-1)
    nbrs.fit(np.stack(df['vec'].to_numpy()))
    return df, nbrs


@st.cache(allow_output_mutation=True)
def load_model():
    clip, _, preproc = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e32')
    return clip, preproc


db, neighbours = load_db()
model, preprocess = load_model()
st.title('Image Search')
uploaded_file = st.file_uploader('Choose an image...', type=['png', 'jpg', 'jpeg', 'webp', 'tiff'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    img = ImageOps.exif_transpose(img)
    with torch.no_grad():
        image = preprocess(img).unsqueeze(0)
        vec = model.encode_image(image).cpu().detach().numpy()

    indices = neighbours.kneighbors(vec.reshape(1, -1), return_distance=False)[0]
    paths = np.hstack(db.loc[indices, ['path']].values)

    for path in paths:
        st.image(path, caption=path)
