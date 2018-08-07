import os
from matplotlib import pyplot
from keras.optimizers import Adam
import keras.callbacks as KC
from keras.utils import multi_gpu_model
import math

from unet import UNet
from dice_coefficient import DiceLossByClass
from images_loader import load_images, save_images
from option_parser import get_option
from data_generator import DataGenerator
from history_checkpoint_callback import HistoryCheckpoint


CLASS_NUM = 1
INPUT_IMAGE_SHAPE = (256, 256, 3)
BATCH_SIZE = 20
EPOCHS = 1000
GPU_NUM = 4

DIR_MODEL = os.path.join('..', 'model')
DIR_INPUTS = os.path.join('..', 'inputs')
DIR_OUTPUTS = os.path.join('..', 'outputs')
DIR_TEACHERS = os.path.join('..', 'teachers_gray')
DIR_TESTS = os.path.join('..', 'predict_data')

File_MODEL = 'segmentation_model.hdf5'


def train(gpu_num=None):
    network = UNet(INPUT_IMAGE_SHAPE, CLASS_NUM)
    model = network.model()
    model.summary()
    if isinstance(gpu_num, int):
        model = multi_gpu_model(model, gpus=gpu_num)
    model.compile(optimizer='adam', loss=DiceLossByClass(INPUT_IMAGE_SHAPE, CLASS_NUM).dice_coef_loss)

    model_filename=os.path.join(DIR_MODEL, File_MODEL)
    callbacks = [ KC.TensorBoard()
                , HistoryCheckpoint(filepath='LearningCurve_{history}.png'
                                    , verbose=1
                                    , period=10
                                   )
                , KC.ModelCheckpoint(filepath=model_filename
                                     , verbose=1
                                     , save_weights_only=True
                                     #, save_best_only=True
                                     , period=10
                                    )
                ]

    print('data generating ...')
    train_generator = DataGenerator(DIR_INPUTS, DIR_TEACHERS, INPUT_IMAGE_SHAPE)
    inputs, teachers = train_generator.generate_data()
    print('... data generated')

    history = model.fit(inputs, teachers, batch_size=BATCH_SIZE, epochs=EPOCHS
                        , shuffle=True, verbose=1, callbacks=callbacks)
    model.save_weights(model_filename)
    plotLearningCurve(history)


def train_with_generator(gpu_num=None):
    network = UNet(INPUT_IMAGE_SHAPE, CLASS_NUM)
    model = network.model()
    if isinstance(gpu_num, int):
        model = multi_gpu_model(model, gpus=gpu_num)
    model.compile(optimizer='adam', loss=DiceLossByClass(INPUT_IMAGE_SHAPE, CLASS_NUM).dice_coef_loss)

    model_filename=os.path.join(DIR_MODEL, File_MODEL)
    callbacks = [ KC.TensorBoard()
                , HistoryCheckpoint(filepath='LearningCurve_{history}.png'
                                    , verbose=1
                                    , period=10
                                   )
                , KC.ModelCheckpoint(filepath=model_filename
                                     , verbose=1
                                     , save_weights_only=True
                                     #, save_best_only=True
                                     , period=10
                                    )
                ]

    train_generator = DataGenerator(DIR_INPUTS, DIR_TEACHERS, INPUT_IMAGE_SHAPE)
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
    model.save_weights(model_filename)
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


def predict(input_dir, gpu_num=None):
    (file_names, inputs) = load_images(input_dir, INPUT_IMAGE_SHAPE)

    network = UNet(INPUT_IMAGE_SHAPE, CLASS_NUM)

    model = network.model()
    if isinstance(gpu_num, int):
        model = multi_gpu_model(model, gpus=gpu_num)
    model.summary()
    model.load_weights(os.path.join(DIR_MODEL, File_MODEL))
    print('predicting ...')
    preds = model.predict(inputs, BATCH_SIZE)
    print('... predicted')

    print('output saveing ...')
    save_images(DIR_OUTPUTS, preds, file_names)
    print('... saved')


if __name__ == '__main__':
    args = get_option(EPOCHS)
    EPOCHS = args.epoch

    if not(os.path.exists(DIR_MODEL)):
        os.mkdir(DIR_MODEL)
    if not(os.path.exists(DIR_OUTPUTS)):
        os.mkdir(DIR_OUTPUTS)

    #train(gpu_num=GPU_NUM)
    #train_with_generator(gpu_num=GPU_NUM)

    #predict(DIR_INPUTS)
    predict(DIR_TESTS, gpu_num=GPU_NUM)
