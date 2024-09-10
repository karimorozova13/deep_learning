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
data_path = './topic05/'
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
     
      # read the image using the opencv library
      imgs = [cv2.imread(p) for p in imgs_path]
      # matplotlib expects img in RGB format but OpenCV provides it in BGR       
      # transform the BGR image into RGB
      imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
       
      # create a figure for display
      fig = plt.figure(figsize=(7, 2))
      grid = ImageGrid(
        fig, 111, 
        nrows_ncols=(1, 4)
      )
      # display the image
      for ax, img in zip(grid, imgs):
        ax.imshow(img)

      fig.suptitle(f'Class {k}, {s.capitalize()} split')
      plt.show()

