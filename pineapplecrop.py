import random
import cv2
import numpy as np
from PIL import Image, ImageDraw
import random

im = Image.open("pineapple.jpg")
width, height = im.size
print(width, height)
#im = cv2.resize(imi, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
#im1 = np.asarray(im)

# Opens a image in RGB mode
# Setting the points for cropped image
for i in range(3000):
    x = random.randint(0, 65)
    y = random.randint(0, 271)
    left = x
    top = y
    right = x+256
    bottom = y+256
    #print(left, top, right, bottom)
    imi = im.crop((left, top, right, bottom))
    name = str(i)
    filename = "%s.jpg" % name
    print(filename)
    imi.save(filename)
