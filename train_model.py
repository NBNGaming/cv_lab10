import torch
import pickle
import open_clip
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

print('Loading model...')
model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')

des_list = []
print('Processing images:')
with torch.no_grad():
    for file in tqdm(glob('val2017/*')):
        image = preprocess(Image.open(file)).unsqueeze(0)
        image_features = model.encode_image(image).cpu().detach().numpy()
        des_list += list(image_features)
des_list = np.array(des_list)

print('Training k-means:')
n = 2048
model = MiniBatchKMeans(n_clusters=n, n_init='auto', verbose=1)
model.fit(des_list)

print('Saving result..')
with open('model.pickle', 'wb') as f:
    pickle.dump((model, n), f)
print('Done!')
