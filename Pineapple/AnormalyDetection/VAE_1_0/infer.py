import glob
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image

IMAGE_SIZE = 256
CHANNEL = 3

json_file = 'ae_pinapple.json'
weight_file = 'ae_weights.h5'
data_path = 'C:/Users/zenkori/Documents/ObjectTracking/data/test/'
#jpg_files = ['ose1.jpg', 'ose2.jpg', 'ose3.jpg', 'ose1256.jpg', 'ose2256.jpg', 'ose3256.jpg', 'pineapple256.jpg']
jpg_files = ['pineapple.jpg']
#jpg_files = ['4.jpg']

json = open(json_file).read()
model = model_from_json(json)
model.load_weights(weight_file)


def infer(img):
    """
    Infer encoded and decoded image from input.
    input: pillow image type:                  img
    output: numpy array like image(uint8)type: decoded_img
            uint8 type:                        err
    """
    ori_img = img_to_array(img)
    ori_img = ori_img.astype('float32') / 255.0
    ori_img = np.reshape(ori_img, (1, IMAGE_SIZE, IMAGE_SIZE, CHANNEL))

    decoded_img = model.predict(ori_img)
    err = np.sum(np.square(decoded_img - ori_img)) / (IMAGE_SIZE * IMAGE_SIZE * CHANNEL)

    decoded_img = decoded_img.reshape(IMAGE_SIZE, IMAGE_SIZE, CHANNEL)
    decoded_img = decoded_img * 255
    decoded_img = decoded_img.astype('uint8')

    return (decoded_img, err)

#
def main():
    print('json:', json_file)
    print('weight:', weight_file)
    model.summary()

    for jpg in jpg_files:
        jpg_base = os.path.splitext(jpg)[0]
        jpg_name = os.path.join(data_path, jpg)
        print(jpg)
        img = Image.open(jpg_name)

        w, h = img.size
        print('h = ', h, ' w = ', w)
        new_w = math.ceil(w / IMAGE_SIZE) * IMAGE_SIZE
        new_h = math.ceil(h / IMAGE_SIZE) * IMAGE_SIZE
    
        img = img.resize((new_w, new_h))
        # numpy はh, w
        heat_map = np.zeros((new_h, new_w), np.float32)
        decoded_concat = np.zeros((new_h, new_w, CHANNEL), np.uint8)

        for x in range(0, new_w, IMAGE_SIZE):
            for y in range(0, new_h, IMAGE_SIZE):
                split_img = img.crop((x, y, x+IMAGE_SIZE, y+IMAGE_SIZE))

                fname = "%s_%d_%d.jpg" % (jpg_base, x, y)
                dst = os.path.join(data_path, fname)
#                split_img.save(dst)

                (decoded_img, err) = infer(split_img)
                print("y = %d, x = %d err = %f" % (y, x, err))
                heat_map[y:y+IMAGE_SIZE, x:x+IMAGE_SIZE] = err

                im = Image.fromarray(decoded_img, 'RGB')
                decoded_concat[y:y+IMAGE_SIZE, x:x+IMAGE_SIZE, :] = decoded_img
                fname = "%s_decoded_%d_%d.jpg" % (jpg_base, x, y)
                dst = os.path.join(data_path, fname)
#                im.save(dst)
        # make heatmap
        heat_map = np.uint8(255 * heat_map) * 10
        hm_img = Image.fromarray(heat_map)
        heat_map_name = os.path.join(data_path, jpg_base + '_hm.png')
        hm_img.save(heat_map_name)
        print("write ", heat_map_name)
        decoded_concat_name = os.path.join(data_path, jpg_base + '_decoded.png')
        decoded_concat_img = Image.fromarray(decoded_concat, 'RGB')
        decoded_concat_img.save(decoded_concat_name)

    print('finish')

if __name__ =="__main__":
    main()

