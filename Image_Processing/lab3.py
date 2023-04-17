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

# Implement Kmeans from scratch
def my_kmeans(I, k):
    assert len(I.shape) == 2, '''Wrong input dimensions! 
            Please make sure you are using a gray-scale image!'''
    X = np.array(I).flatten()
    clusters = np.random.choice(X, k, replace=False)
    max_iter = 250
    for iter in range(max_iter):
        L = np.argmin(np.abs(X.reshape((-1, 1)) - clusters.reshape((1, -1))), axis=1)
        for i in range(k):
            clusters[i] = np.mean(X[L == i])
    L = L.reshape(I.shape)
    return clusters, L 

clusters, L = my_kmeans(gray_img, k)
plt.subplot(2, 3, 3)
plt.imshow(L)

# Implement Kmeans from scratch
def my_kmeans_rgb(I, k):
    assert len(I.shape) == 3, '''Wrong input dimensions!
      Please make sure you are using a RGB image!'''
    
    pixels = np.reshape(I, (I.shape[0]*I.shape[1], 3))
    centers = np.random.choice(pixels.shape[0], k, replace=False)
    clusters = pixels[centers]
    
    # Assign each pixel to its nearest cluster center
    L = np.zeros(pixels.shape[0])
    for i, p in enumerate(pixels):
        L[i] = np.argmin(np.linalg.norm(clusters - p, axis=1))
    
    # Iterate until convergence (or max iterations)
    max_iters = 100
    for iter in range(max_iters):
        # Update cluster centers
        for j in range(k):
            clusters[j] = np.mean(pixels[L==j], axis=0)
        
        # Reassign pixels to nearest cluster center
        for i, p in enumerate(pixels):
            L[i] = np.argmin(np.linalg.norm(clusters - p, axis=1))
    L = np.reshape(L, (I.shape[0], I.shape[1]))
    return clusters, L

clusters, L = my_kmeans_rgb(img, k)
plt.subplot(2, 3, 6)
plt.imshow(L)

plt.show()