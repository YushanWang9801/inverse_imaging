import numpy as np
import cv2
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

def display_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    
def to_gray(img):
    '''
    Convert and RGB image into gray scale
    :param img: RBG image with size HxWx3
    :return: gray scaled image of size HxW
    '''
    assert len(img.shape) == 3, f'Wrong input image dimensions, we expected an input of size HxWxC instead we got {img.shape}'
    h,w,c = img.shape
    # Modify this part to convert the image onto gray scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    assert gray_img.shape == (h,w), 'Wrong spacial dimensions, your method should only change the channel dimension'
    return gray_img

def flip_diagonal(img):
    return cv2.flip(cv2.transpose(img), 1)

def segment_four_parts(img):
    height, width, channels = img.shape
    center_x, center_y = width // 2, height // 2
    top_left = img[0:center_y, 0:center_x]
    top_right = img[0:center_y, center_x:width]
    bottom_left = img[center_y:height, 0:center_x]
    bottom_right = img[center_y:height, center_x:width]
    return (top_left, top_right, bottom_left, bottom_right)

def scale_down(img):
    height, width, channels = img.shape
    scaled_img = cv2.resize(img, (width // 2, height // 2), 
                            interpolation=cv2.INTER_AREA)
    return scaled_img

# Display image
img = cv2.imread("./images/apple1.jpg")
# display_image(img)

# Convert image to gray scale
gray_image = to_gray(img)
# display_image(gray_image)

# Flip the image across its diagonal.
flip_diag_image = flip_diagonal(img)
# display_image(flip_diag_image)

# Split the image into four equal parts and display each of them
four_parts = segment_four_parts(img)
# for part in four_parts:
#     display_image(part)

# Use openCV python routines to scale down the image and 4
# plot them side by side with the full scaled image.
scaled_down_image = scale_down(img)
# display_image(scaled_down_image)

