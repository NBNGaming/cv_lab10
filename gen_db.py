import torch
import open_clip
import pandas as pd
from glob import glob
from tqdm import tqdm
from PIL import Image

print('Loading model...')
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e32')

paths = []
vectors = []
with torch.no_grad():
    for path in tqdm(glob('voc2012/*'), 'Processing images'):
        image = preprocess(Image.open(path)).unsqueeze(0)
        vec = model.encode_image(image).cpu().detach().numpy()[0]
        paths.append(path)
        vectors.append(vec)

print('Saving db..')
df = pd.DataFrame({'path': paths, 'vec': vectors})
df.to_pickle('db.pickle')
print('Done!')
