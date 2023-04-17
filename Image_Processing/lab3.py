import numpy as np
import copy
import cv2
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

# K-means Segmentation
raw_img = cv2.imread("./images/sample_image.jpg") # change file name to load different images
raw_gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
img = raw_img.astype(np.float32) / 255.
gray_img = raw_gray_img.astype(np.float32) / 255.
plt.subplot(2, 3, 1)
plt.imshow(img)
plt.subplot(2, 3, 2)
plt.imshow(gray_img, "gray")

# Using sklearn
from sklearn.cluster import KMeans
k = 2
pixels = gray_img.reshape((-1, 1))
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(pixels)

labels = kmeans.predict(pixels)
L, clusters = labels.reshape(gray_img.shape), kmeans.cluster_centers_
plt.subplot(2, 3, 4)
plt.imshow(L, "gray")

# Now on RGB images
pixels = img.reshape((-1, 3))
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(pixels)
labels = kmeans.predict(pixels)
labels = labels.reshape((img.shape[0], img.shape[1]))
plt.subplot(2, 3, 5)
plt.imshow(labels)



plt.show()