import os
import glob
import numpy as np
from PIL import Image
from keras.optimizers import Adam
import keras.backend as Kbackend

from unet import UNet


def dice_coef(y_true, y_pred):
    y_true = Kbackend.flatten(y_true)
    y_pred = Kbackend.flatten(y_pred)
    intersection = Kbackend.sum(y_true * y_pred)
    return 2.0 * intersection / (Kbackend.sum(y_true) + Kbackend.sum(y_pred) + 1)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def load_images(dir_name):
    files = glob.glob(os.path.join(dir_name, '*.png'))
    files.sort()
    images = np.zeros((len(files), 572, 572, 1), 'float32')
    for i, file in enumerate(files):
        srcImg = Image.open(file)
        distImg = srcImg.convert('L')
        imgArray = np.asarray(distImg)
        imgArray = np.reshape(imgArray, (572, 572, 1))
        images[i] = imgArray / 255
    return (files, images)


def train():
    (_, inputs) = load_images(os.path.join('..', 'Inputs'))
    (_, teachers) = load_images(os.path.join('..', 'Teachers'))

    network = UNet()
    model = network.model()
    model.compile(Adam(), dice_coef_loss, [dice_coef], )

    model.fit(inputs, teachers, 5, 100, 1)
    model.save_weights(os.path.join('..', 'Model', 'cat_detect_model.hdf5'))


def predict():
    (file_names, inputs) = load_images(os.path.join('..', 'Inputs'))
    network = UNet()
    model = network.model()
    model.load_weights(os.path.join('..', 'Model', 'cat_detect_model.hdf5'))
    preds = model.predict(inputs, 5)

    for pred, file_name in enumerate(preds, file_names):
        name = os.path.basename(file_name)
        distImg = Image.fromarray(pred * 255)
        distImg.save(os.path.join('..', 'Outputs', name))


if __name__ == '__main__':
    train()
    predict()
