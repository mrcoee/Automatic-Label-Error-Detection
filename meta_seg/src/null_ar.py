import numpy as np
from PIL import Image 
import os

root_dir = '/home/reese/coco_eval/'

d_list = []
for img_fn in os.listdir(root_dir + 'inference/'):
    img = Image.open(root_dir + 'inference/' + img_fn)
    if len(np.unique(np.array(img))) == 1:
        d_list.append(img_fn)

print(d_list)
for subdir in ['inference/', 'masks_perturbed/', 'masks_unperturbed/', 'rgb/']:
    for img_n in d_list:
        os.remove(root_dir + subdir + img_n) 

for img_n in d_list:
    os.remove((root_dir + 'npy/' + img_n).replace('png', 'npy')) 