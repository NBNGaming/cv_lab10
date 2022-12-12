import cv2
import pickle
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm


def img_to_vec(gray, estimator, n_clusters):
    sift = cv2.SIFT_create()
    des = sift.detectAndCompute(gray, None)[1]
    if des is None:
        return None
    classes = estimator.predict(des)
    hist = np.histogram(classes, n_clusters, density=True)[0]
    return hist


if __name__ == '__main__':
    with open('model.pickle', 'rb') as f:
        model, n = pickle.load(f)
    paths = []
    vectors = []
    print('Processing images:')
    for path in tqdm(glob('voc2012/*')):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        vec = img_to_vec(img, model, n)
        if vec is None:
            continue
        paths.append(path)
        vectors.append(vec)

    print('Saving db..')
    df = pd.DataFrame({'path': paths, 'vec': vectors})
    df.to_pickle('db.pickle')
    print('Done!')
