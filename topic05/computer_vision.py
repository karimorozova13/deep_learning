# %%
import os
import random
from collections import defaultdict
from tqdm.autonotebook import tqdm

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import cv2
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F

from sklearn.metrics import balanced_accuracy_score

import warnings
# filter warnings
warnings.filterwarnings('ignore')

# %%
data_path = './'
splits = ['train', 'test']

# %%
# Display images examples

# iterate over train and test folders
for s in ['train']:
  files = [f for f in os.listdir(f"{data_path}{s}_signs") if f.endswith('.jpg')]

  print(f'{len(files)} images in {s}')
   
  # for each image, create a list of the type [class, filename]
  files = [f.split('_', 1) for f in files]
   
  # group the data by class
  files_by_sign = defaultdict(list)
  for k, v in files:
    files_by_sign[k].append(v)
   
  # take random 4 images of each class  
  for k, v in sorted(files_by_sign.items()):
    print(f'Number of examples for class {k}:', len(v))
     
    # display several examples of images from the training sample   
    if s == 'train':        
      random.seed(42)
      imgs_path = random.sample(v, 4)
      imgs_path = [os.path.join(data_path, f'{s}_signs/{k}_{p}') for p in imgs_path]
      print(imgs_path)
      # read the image using the opencv library
      imgs = [cv2.imread(p) for p in imgs_path]
      # matplotlib expects img in RGB format but OpenCV provides it in BGR       
      # transform the BGR image into RGB
      imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
       
      # create a figure for display
      fig = plt.figure(figsize=(7, 2))
      grid = ImageGrid(fig, 111, nrows_ncols=(1, 4))
      # display the image
      for ax, img in zip(grid, imgs):
          ax.imshow(img)

      fig.suptitle(f'Class {k}, {s.capitalize()} split')
      plt.show()

# %%
orig_img = Image.fromarray(img)

def plot_examples(transformed_imgs:list, col_titles:list, cmap=None):
    
    n_cols = len(transformed_imgs) +1
    fig_size_x = 3 + len(transformed_imgs) * 1.5
    fig, axs = plt.subplots(1, n_cols, figsize=(fig_size_x,2))
    
    axs[0].imshow(orig_img)
    axs[0].set_title('original image')
    
    for i in range(len(transformed_imgs)):
        axs[i+1].imshow(transformed_imgs[i], cmap=cmap)
        axs[i+1].set_title(col_titles[i])
    
    plt.tight_layout()
    plt.show()

# %%
# resize

resized_imgs = [T.Resize(size=size)(orig_img) for size in [32,128]]
plot_examples(resized_imgs, ['32x32', '128x128'])

# %%
# gray scale

gray_img = T.Grayscale()(orig_img)
plot_examples([gray_img], ["Gray"], 'gray')

# %%
# normalize

normalized_img = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(T.ToTensor()(orig_img)) 
normalized_img = [T.ToPILImage()(normalized_img)]
plot_examples(normalized_img, ["Standard normalize"])

# %%
# random rotation

rotated_imgs = [T.RandomRotation(degrees=d)(orig_img) for d in range(50,151,50)]
plot_examples(rotated_imgs, ["Rotation 50","Rotation 100","Rotation 150"])

# %%
# center crop

center_crops = [T.CenterCrop(size=size)(orig_img) for size in (1280,640, 320)]
plot_examples(center_crops,['1280x1280','640x640','320x320'])

# %%
# random crop

random_crops = [T.RandomCrop(size=size)(orig_img) for size in (832,704, 256)]
plot_examples(random_crops,['832x832','704x704','256x256'])

# %%
# Gaussian Blur

downsized_img = T.Resize(size=512)(orig_img)
blurred_imgs = [T.GaussianBlur(kernel_size=(51, 91), sigma=sigma)(downsized_img) for sigma in (3,7)]
plot_examples(blurred_imgs, ['sigma=3', 'sigma=7'])

# %%
# Gaussian noise

def add_noise(inputs,noise_factor=0.3):
    noisy = inputs+torch.randn_like(inputs) * noise_factor
    noisy = torch.clip(noisy,0.,1.)
    return noisy
    
noise_imgs = [add_noise(T.ToTensor()(orig_img),noise_factor) for noise_factor in (0.3,0.6,0.9)]
noise_imgs = [T.ToPILImage()(noise_img) for noise_img in noise_imgs]
plot_examples(noise_imgs, ["noise_factor=0.3","noise_factor=0.6","noise_factor=0.9"])

# %%
# Random Blocks

def add_random_boxes(img,n_k,size=32):
    h,w = size,size
    img = np.asarray(img)
    img_size = img.shape[1]
    boxes = []
    for k in range(n_k):
        y,x = np.random.randint(0,img_size-w,(2,))
        img[y:y+h,x:x+w] = 0
        boxes.append((x,y,h,w))
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    return img

blocks_imgs = [add_random_boxes(orig_img,n_k=i, size=128) for i in (10,20)]
plot_examples(blocks_imgs, ["10 black boxes","20 black boxes"])

