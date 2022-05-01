from importlib.resources import path
import os 
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def load_imgs(path):
    #path = '/mnt/c/Users/schuy/Pictures/yalefaces/yalefaces'    
    img_fnames  = [os.path.join(path, x) for x in os.listdir(path)]
    imgs = [Image.open(x) for x in img_fnames]    
    return imgs

def plot_images(imgs, fout, n=5, m=5):
    #fsave = '/mnt/c/Users/schuy/Documents/plots/face_imgs.png'
    fig,ax = plt.subplots(n, m)
    for row in range(n):
        for col in range(m):
            ax[row,col].imshow(imgs[row+col])
    fig.savefig(fout)    

path = '/mnt/c/Users/schuy/Pictures/yalefaces/yalefaces'
imgs = load_images(path)
print()