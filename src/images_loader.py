import os
import glob
import numpy as np
from PIL import Image


def load_images(dir_name, size):
    files = glob.glob(os.path.join(dir_name, '*.png'))
    files.sort()
    images = np.zeros((len(files), size, size, 1), 'float32')
    for i, file in enumerate(files):
        srcImg = Image.open(file)
        distImg = srcImg.convert('L')
        imgArray = np.asarray(distImg)
        imgArray = np.reshape(
            imgArray, (size, size, 1))
        images[i] = imgArray / 255
    return (files, images)


def save_images(dir_name, image_data_list, file_name_list):
    for _, (image_data, file_name) in enumerate(zip(image_data_list, file_name_list)):
        name = os.path.basename(file_name)
        (w, h, _) = image_data.shape
        image_data = np.reshape(image_data, (w, h))
        distImg = Image.fromarray(image_data * 255)
        distImg = distImg.convert('RGB')
        save_path = os.path.join(dir_name, name)
        distImg.save(save_path, "png")
        print(save_path)
