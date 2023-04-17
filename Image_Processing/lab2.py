import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image(img):
    plt.imshow(img)
    plt.colorbar()
    plt.show()

img = cv2.imread('./images/sample_image.jpg', cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_gray = img_gray.astype(np.double) / 255

display_image(img_gray)

# Next, we take a sample of this image
row = img_gray.shape[1] // 2
x = img_gray[row, :]
plt.plot(x)
plt.title('Grey-level profile at ' + str(row))
plt.show()