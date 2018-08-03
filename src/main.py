import os
from matplotlib import pyplot
from keras.optimizers import Adam
import keras.callbacks as KC
import math

from unet import UNet
from dice_coefficient import DiceLossByClass
from images_loader import load_images, save_images
from option_parser import get_option
from data_generator import DataGenerator


CLASS_NUM = 3
INPUT_IMAGE_SHAPE = (256, 256, CLASS_NUM)
BATCH_SIZE = 20
EPOCHS = 100

DIR_MODEL = os.path.join('..', 'model')
DIR_INPUTS = os.path.join('..', 'inputs')
DIR_OUTPUTS = os.path.join('..', 'outputs')
DIR_TEACHERS = os.path.join('..', 'teachers')
DIR_TESTS = os.path.join('..', 'TestData')

File_MODEL = 'segmentation_model.hdf5'


def train():
    print('input data loading ...', )
    (_, inputs) = load_images(DIR_INPUTS, INPUT_IMAGE_SHAPE)
    print('... loaded .', )
    print('teacher data loading ...', )
    (_, teachers) = load_images(DIR_TEACHERS, INPUT_IMAGE_SHAPE)
    print('... loaded .', )

    network = UNet(INPUT_IMAGE_SHAPE, CLASS_NUM)
    model = network.model()
    model.compile(optimizer='adam', loss=DiceLossByClass(INPUT_IMAGE_SHAPE, CLASS_NUM).dice_coef_loss)

    callbacks = [ KC.TensorBoard()
                ]

    history = model.fit(inputs, teachers, batch_size=BATCH_SIZE, epochs=EPOCHS
                        , shuffle=True, verbose=1, callbacks=callbacks)
    model.save_weights(os.path.join(DIR_MODEL, File_MODEL))
    plotLearningCurve(history)


def train_with_generator():
    network = UNet(INPUT_IMAGE_SHAPE, CLASS_NUM)
    model = network.model()
    model.compile(optimizer='adam', loss=DiceLossByClass().dice_coef_loss)

    callbacks = [ KC.TensorBoard()
                ]

    train_generator = DataGenerator(DIR_INPUTS, DIR_INPUTS, INPUT_IMAGE_SHAPE)
    train_data_num = train_generator.data_size()


    print('fix ...')
    his = model.fit_generator(train_generator.generator(batch_size=BATCH_SIZE)
                              , steps_per_epoch=math.ceil(train_data_num / BATCH_SIZE)
                              , epochs=EPOCHS
                              , verbose=1
                              , use_multiprocessing=True
                              , callbacks=callbacks
                              #, validation_data=valid_generator
                              #, validation_steps=math.ceil(valid_data_num / BATCH_SIZE)
                             )
    print('model saveing ...')
    model.save_weights(os.path.join(DIR_MODEL, File_MODEL))
    print('... saved')
    plotLearningCurve(his)


def plotLearningCurve(history):
    """ saveLearningCurve """
    x = range(EPOCHS)
    pyplot.plot(x, history.history['loss'], label="loss")
    pyplot.title("loss")
    pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    lc_name = 'LearningCurve'
    pyplot.savefig(lc_name + '.png')
    pyplot.close()


def predict(input_dir):
    (file_names, inputs) = load_images(input_dir, INPUT_IMAGE_SHAPE)

    network = UNet(INPUT_IMAGE_SHAPE, CLASS_NUM)
    model = network.model()
    model.load_weights(os.path.join(DIR_MODEL, File_MODEL))
    preds = model.predict(inputs, BATCH_SIZE)

    save_images(DIR_OUTPUTS, preds, file_names)


if __name__ == '__main__':
    args = get_option(EPOCHS)
    EPOCHS = args.epoch

    if not(os.path.exists(DIR_MODEL)):
        os.mkdir(DIR_MODEL)
    if not(os.path.exists(DIR_OUTPUTS)):
        os.mkdir(DIR_OUTPUTS)

    #train()
    train_with_generator()

    #predict(DIR_INPUTS)
    #predict(DIR_TESTS)
