import os
from matplotlib import pyplot
from keras.optimizers import Adam

from unet import UNet
from dice_coefficient import dice_coef_loss, dice_coef
from images_loader import load_images, save_images


INPUT_IMAGE_SIZE = 128
BATCH_SIZE = 5
EPOCHS = 10

DIR_MODEL = os.path.join('..', 'Model')
DIR_INPUTS = os.path.join('..', 'Inputs')
DIR_OUTPUTS = os.path.join('..', 'Outputs')
DIR_TEACHERS = os.path.join('..', 'Teachers')
DIR_TESTS = os.path.join('..', 'TestData')

File_MODEL = 'detect_cat_model.hdf5'


def train():
    (_, inputs) = load_images(DIR_INPUTS, INPUT_IMAGE_SIZE)
    (_, teachers) = load_images(DIR_TEACHERS, INPUT_IMAGE_SIZE)

    network = UNet(INPUT_IMAGE_SIZE)
    model = network.model()
    model.compile(optimizer='adam', loss=dice_coef_loss, metrics=[dice_coef])

    history = model.fit(
        inputs, teachers, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
    model.save_weights(os.path.join(DIR_MODEL, File_MODEL))
    plotLearningCurve(history)


def plotLearningCurve(history):
    x = range(EPOCHS)
    pyplot.plot(x, history.history['loss'], label="loss")
    pyplot.title("loss")
    pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    pyplot.show()


def predict(input_dir):
    (file_names, inputs) = load_images(input_dir, INPUT_IMAGE_SIZE)

    network = UNet(INPUT_IMAGE_SIZE)
    model = network.model()
    model.load_weights(os.path.join(DIR_MODEL, File_MODEL))
    preds = model.predict(inputs, BATCH_SIZE)

    save_images(DIR_OUTPUTS, preds, file_names)


if __name__ == '__main__':
    if not(os.path.exists(DIR_MODEL)):
        os.mkdir(DIR_MODEL)
    if not(os.path.exists(DIR_OUTPUTS)):
        os.mkdir(DIR_OUTPUTS)

    train()

    predict(DIR_INPUTS)
    predict(DIR_TESTS)
