import os
import glob
import numpy as np
from PIL import Image
import cv2


def load_images(dir_name, image_shape):
    files = glob.glob(os.path.join(dir_name, '*.png'))
    files.sort()
    h, w, ch = image_shape
    images = []
    print ('load_images : ', len(files))
    for i, file in enumerate(files):
        #src_img = Image.open(file)
        #dist_img = src_img.convert('L')
        #img_array = np.asarray(dist_img)
        #img_array = np.reshape(img_array, image_shape)
        #images.append(img_array / 255)
        src_img = cv2.imread(file)
        dist_img = src_img
        if ch == 1:
            dist_img = cv2.cvtColor(dist_img, cv2.COLOR_BGR2GRAY)
        dist_img = cv2.resize(dist_img, (h, w))
        norm_img = dist_img / 255
        images.append(norm_img)
        if i % 500 == 0:
            print('load_images loaded ', i)
    return (files, np.array(images, dtype=np.float32))


def save_images(dir_name, image_data_list, file_name_list):
    for _, (image_data, file_name) in enumerate(zip(image_data_list, file_name_list)):
        name = os.path.basename(file_name)
        #(w, h, _) = image_data.shape
        #image_data = np.reshape(image_data, (w, h))
        #distImg = Image.fromarray(image_data * 255)
        #distImg = distImg.convert('RGB')
        save_path = os.path.join(dir_name, name)
        #distImg.save(save_path, "png")
        cv2.imwrite(image_data * 255, save_path)
        print('saved : ' , save_path)
