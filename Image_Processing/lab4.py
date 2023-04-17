import numpy as np
import copy
import cv2
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import random
from PIL import Image

img = cv2.imread('./images/sample_image.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
# plt.imshow(gray_img, 'gray')

class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y

def regionGrow(img, seeds, thresh, p = 1):
    assert len(img.shape) == 2, f'''Wrong input image dimensions, 
    we expected an input of size HxW instead we got {img.shape}'''

    H, W = img.shape
    segmented_img, Q = np.zeros((H, W)), []
    for seed in seeds:
        Q.append(seed)
    while len(Q) > 0:
        current = Q.pop(0)
        x, y = current.x, current.y
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                if p == 1 and abs(i) == abs(j):
                    continue
                if x + i < 0 or y + j < 0 or x + i >= H or y + j >= W:
                    continue
                if segmented_img[x + i][y + j] != 0:
                    continue
                diff = abs(int(img[x][y]) - int(img[x + i][y + j]))
                if diff <= thresh:
                    Q.append(Point(x + i, y + j))
                    segmented_img[x + i][y + j] = 255

    assert segmented_img.shape == img.shape, f'''Wrong output image dimensions, 
    we expected to be same size as input {img.shape} instead we got
      {segmented_img.shape}'''
    
    return segmented_img

seg_img = regionGrow(gray_img, [Point(50,50)], 0.2)
# Show Original and segmented image
fig, axis = plt.subplots(1, 3, figsize=(15,10), sharey=True)
fig.subplots_adjust(wspace=0.1, hspace=0.05)
axis[0].imshow(img)
axis[0].set_title('Original RGB image')
axis[1].imshow(gray_img, 'gray')
axis[1].set_title('Gray scale image')
axis[2].imshow(seg_img, 'gray')
axis[2].set_title('Segmented image')
plt.show()



