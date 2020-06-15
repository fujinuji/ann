from sklearn.datasets import load_digits

import cv2

# load the input image and resize it to the target spatial dimensions
from Utils import load_data

width = 64
height = 64
image = cv2.imread("images\sad\sad1-00.png")
print(image)

output = image.copy()
image = cv2.resize(image, (width, height))
# scale the pixel values to [0, 1]
print(image)
image = image.astype("float") / 255.0

# when working with a CNN: don't flatten the image, simply add the batch dimension
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

load_data()