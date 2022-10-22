from importlib.resources import path
import os
from sqlite3 import paramstyle 
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import glob

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

def partition_dataset_per_person(path):
    img_fnames = [os.path.join(path, x) for x in os.listdir(path)]
    split = [fname.split('/')[-1].split('_')[0] for fname in img_fnames]
    unique_subject_names = []
    for fname in split:
        if fname not in unique_subject_names:
            unique_subject_names.append(fname)
    glob_dict = {'{}'.format(subject) : glob.glob(os.path.join(path,'{}*'.format(subject))) for subject in unique_subject_names}
    test = {'{}'.format(subject): subject for subject in unique_subject_names}
    return glob_dict

def plt_img(img, fout):
    fig,ax = plt.subplots()
    ax.imshow(img)
    fig.savefig(fout)

def PCA(B):
    S = np.dot(B,B.T)
    eigval, eigvec = np.linalg.eig(S)
    U = np.matrix(B.T)*np.matrix(np.real(eigvec).T)
    return U

path = '/mnt/c/Users/schuy/Pictures/yalefaces/yalefaces'
imgs = [np.array(img) for img in load_imgs(path)]
dim0, dim1 = imgs[0].shape[0], imgs[0].shape[1] 
partition_dataset = partition_dataset_per_person(path)

B = np.vstack([img.flatten() for img in imgs])
mean_img = np.mean(B, axis=0)
print(B.shape)
B_x = np.vstack([b - mean_img for b in B])

U = PCA(B_x)
print('dimU', U.shape)


'''
for k in range(5):
    img = U[:,k].reshape(dim0,dim1)
    fout = '/mnt/c/Users/schuy/Documents/plots/eigface_mean_centered{}.png'.format(k)
    plt_img(img, fout)
'''

test_idxs = np.random.choice(np.arange(len(imgs)), size=5)
train_idxs = [i for i in range(len(imgs)) if i not in test_idxs]
T = np.vstack([imgs[i].flatten() for i in test_idxs])
B_train = np.vstack([imgs[i].flatten() for i in train_idxs])

W_train = np.asarray(np.matrix(B_train)*np.matrix(U[:,:15]))
W_test = np.asarray(np.matrix(T)*np.matrix(U[:,:15]))

print('W_shapes', W_train.shape, W_test.shape) 

d = lambda u,v: np.sum((u-v)**2)**.5
idx_min = np.argmin(d(W_test[0], W_train))
print(idx_min, W_train.shape)

fig,ax = plt.subplots(1,2)
ax[0].imshow(T[0].reshape(dim0,dim1))
ax[0].set_title('test_img')
ax[1].imshow(B_train[idx_min].reshape(dim0,dim1))
ax[1].set_title('IDed match')
fout = '/mnt/c/Users/schuy/Documents/plots/first_match_attempt.png'
fig.savefig(fout)