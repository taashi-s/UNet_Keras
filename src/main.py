import os
import glob
import numpy as np
from PIL import Image
from keras.optimizers import Adam
import keras.backend as Kbackend

from unet import UNet

INPUT_IMAGE_SIZE = 128


def dice_coef(y_true, y_pred):
    y_true = Kbackend.flatten(y_true)
    y_pred = Kbackend.flatten(y_pred)
    intersection = Kbackend.sum(y_true * y_pred)
    return (2.0 * intersection + 1) / (Kbackend.sum(y_true) + Kbackend.sum(y_pred) + 1)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def load_images(dir_name):
    files = glob.glob(os.path.join(dir_name, '*.png'))
    files.sort()
    images = np.zeros((len(files), INPUT_IMAGE_SIZE,
                       INPUT_IMAGE_SIZE, 1), 'float32')
    for i, file in enumerate(files):
        srcImg = Image.open(file)
        distImg = srcImg.convert('L')
        imgArray = np.asarray(distImg)
        imgArray = np.reshape(
            imgArray, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 1))
        images[i] = imgArray / 255
    return (files, images)


def train():
    (_, inputs) = load_images(os.path.join('..', 'Inputs'))
    (_, teachers) = load_images(os.path.join('..', 'Teachers'))

    network = UNet(INPUT_IMAGE_SIZE)
    model = network.model()
    model.compile(optimizer='adam', loss=dice_coef_loss, metrics=[dice_coef])

    model.fit(inputs, teachers, batch_size=1, epochs=20, verbose=2)
    model.save_weights(os.path.join('..', 'Model', 'cat_detect_model.hdf5'))


def predict():
    (file_names, inputs) = load_images(os.path.join('..', 'Inputs'))
    network = UNet(INPUT_IMAGE_SIZE)
    model = network.model()
    model.load_weights(os.path.join('..', 'Model', 'cat_detect_model.hdf5'))
    preds = model.predict(inputs, 5)

    for _, (pred, file_name) in enumerate(zip(preds, file_names)):
        name = os.path.basename(file_name)
        (w, h, _) = pred.shape
        pred = np.reshape(pred, (w, h))
        distImg = Image.fromarray(pred * 255)
        distImg = distImg.convert('RGB')
        save_path = os.path.join('..', 'Outputs', name)
        distImg.save(save_path, "png")
        print(save_path)


if __name__ == '__main__':
    train()
    predict()
