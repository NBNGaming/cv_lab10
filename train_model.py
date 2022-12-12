import cv2
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

des_list = []
print('Processing images:')
for file in tqdm(glob('val2017/*')):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    des_list += list(des)
des_list = np.array(des_list)

print('Training k-means:')
n = 2048
model = MiniBatchKMeans(n_clusters=n, n_init='auto', verbose=1)
model.fit(des_list)

print('Saving result..')
with open('model.pickle', 'wb') as f:
    pickle.dump((model, n), f)
print('Done!')
