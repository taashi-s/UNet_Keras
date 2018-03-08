import os
import glob
import numpy as np
from PIL import Image
from matplotlib import pyplot
from keras.optimizers import Adam
import keras.backend as Kbackend

from unet import UNet

INPUT_IMAGE_SIZE = 128
TEACHER_IMAGE_SIZE = 36
BATCH_SIZE = 5
EPOCHS = 20000

DIR_MODEL = os.path.join('..', 'Model')
DIR_OUTPUTS = os.path.join('..', 'Outputs')


def dice_coef(y_true, y_pred):
    y_true = Kbackend.flatten(y_true)
    y_pred = Kbackend.flatten(y_pred)
    intersection = Kbackend.sum(y_true * y_pred)
    return (2.0 * intersection + 1) / (Kbackend.sum(y_true) + Kbackend.sum(y_pred) + 1)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


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


def train():
    (_, inputs) = load_images(os.path.join('..', 'Inputs'), INPUT_IMAGE_SIZE)
    (_, teachers) = load_images(os.path.join('..', 'Teachers'), TEACHER_IMAGE_SIZE)

    network = UNet(INPUT_IMAGE_SIZE)
    model = network.model()
    model.compile(optimizer='adam', loss=dice_coef_loss, metrics=[dice_coef])

    history = model.fit(
        inputs, teachers, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2)
    model.save_weights(os.path.join(DIR_MODEL, 'cat_detect_model.hdf5'))

    x = range(EPOCHS)
#    plt.plot(x, stack.history['acc'], label="acc")
#    plt.title("accuracy")
#    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#    plt.show()

    pyplot.plot(x, history.history['loss'], label="loss")
    pyplot.title("loss")
    pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    pyplot.show()


def predict():
    (file_names, inputs) = load_images(
        os.path.join('..', 'Inputs'), INPUT_IMAGE_SIZE)
    network = UNet(INPUT_IMAGE_SIZE)
    model = network.model()
    model.load_weights(os.path.join(DIR_MODEL, 'cat_detect_model.hdf5'))
    preds = model.predict(inputs, 5)

    for _, (pred, file_name) in enumerate(zip(preds, file_names)):
        name = os.path.basename(file_name)
        (w, h, _) = pred.shape
        pred = np.reshape(pred, (w, h))
        distImg = Image.fromarray(pred * 255)
        distImg = distImg.convert('RGB')
        save_path = os.path.join(DIR_OUTPUTS, name)
        distImg.save(save_path, "png")
        print(save_path)


if __name__ == '__main__':
    os.mkdir(DIR_MODEL)
    os.mkdir(DIR_OUTPUTS)
    train()
    predict()
