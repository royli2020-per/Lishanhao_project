import random
import cv2
import numpy as np
from PIL import Image, ImageDraw
import random
im = Image.open("pineapple.jpg")
width, height = im.size
print(im)
print(width, height)
im = np.asarray(im)
im = cv2.resize(im, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
im1 = np.asarray(im)
im = Image.fromarray(im)
im.save("pineapple256.jpg")
