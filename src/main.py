import os
import numpy as np
from matplotlib import pyplot
from keras.optimizers import Adam
import keras.callbacks as KC
from keras.utils import multi_gpu_model, plot_model
import math

from unet import UNet
from dice_coefficient import DiceLossByClass
from images_loader import load_images, save_images
from option_parser import get_option
from data_generator import DataGenerator
from history_checkpoint_callback import HistoryCheckpoint


CLASS_NUM = 3
PADDING = 0
INPUT_IMAGE_SHAPE = (256 + (PADDING * 2), 256 + (PADDING * 2), 3)
BATCH_SIZE = 80 # 45
EPOCHS = 1000
GPU_NUM = 4


DIR_BASE = os.path.join('.', '..')
DIR_MODEL = os.path.join(DIR_BASE, 'model')
DIR_TRAIN_INPUTS = os.path.join(DIR_BASE, 'inputs')
DIR_TRAIN_TEACHERS = os.path.join(DIR_BASE, 'teachers')
DIR_VALID_INPUTS = os.path.join(DIR_BASE, 'valid_inputs')
DIR_VALID_TEACHERS = os.path.join(DIR_BASE, 'valid_teachers')
DIR_OUTPUTS = os.path.join(DIR_BASE, 'outputs')
DIR_TEST = os.path.join(DIR_BASE, 'predict_data')
DIR_PREDICTS = os.path.join(DIR_BASE, 'predict_data')
#DIR_PREDICTS = os.path.join(DIR_BASE, 'predict_data_new')

File_MODEL = 'segmentation_model.hdf5'
#File_MODEL = 'segmentation_model_11520_all.hdf5'

def train(gpu_num=None, with_generator=False, load_model=False, show_info=True):
    print('network creating ... ')#, end='', flush=True)
    network = UNet(INPUT_IMAGE_SHAPE, CLASS_NUM)
    print('... created')

    model = network.model()
    if show_info:
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
                                     , save_best_only=True
                                     , period=10
                                    )
                ]

    if load_model:
        print('loading weghts ... ', end='', flush=True)
        model.load_weights(model_filename)
        print('... loaded') 

    print('data generating ...', end='', flush=True)
    train_generator = DataGenerator(DIR_TRAIN_INPUTS, DIR_TRAIN_TEACHERS, INPUT_IMAGE_SHAPE
                                    , include_padding=(PADDING, PADDING))
    valid_generator = DataGenerator(DIR_VALID_INPUTS, DIR_VALID_TEACHERS, INPUT_IMAGE_SHAPE
                                    , include_padding=(PADDING, PADDING))
    print('... created')

    if with_generator:
        train_data_num = train_generator.data_size()
        history = model.fit_generator(train_generator.generator(batch_size=BATCH_SIZE)
                                      , steps_per_epoch=math.ceil(train_data_num / BATCH_SIZE)
                                      , epochs=EPOCHS
                                      , verbose=1
                                      , use_multiprocessing=True
                                      , callbacks=callbacks
                                      , validation_data=valid_generator.generator(batch_size=BATCH_SIZE)
                                      , validation_steps=math.ceil(valid_data_num / BATCH_SIZE)
                                     )
    else:
        print('data generateing ... ') #, end='', flush=True)
        inputs, teachers = train_generator.generate_data()
        valid_data = valid_generator.generate_data()
        print('... generated')
        history = model.fit(inputs, teachers, batch_size=BATCH_SIZE, epochs=EPOCHS
                            , validation_data=valid_data
                            , shuffle=True, verbose=1, callbacks=callbacks)

    print('model saveing ... ', end='', flush=True)
    model.save_weights(model_filename)
    print('... saved')
    print('learning_curve saveing ... ', end='', flush=True)
    save_learning_curve(history)
    print('... saved')


def save_learning_curve(history):
    """ saveLearningCurve """
    x = range(EPOCHS)
    pyplot.plot(x, history.history['loss'], label="loss")
    pyplot.title("loss")
    pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    lc_name = 'LearningCurve'
    pyplot.savefig(lc_name + '.png')
    pyplot.close()


def predict(input_dir, gpu_num=None):
    h, w, c = INPUT_IMAGE_SHAPE
    org_h, org_w = h - (PADDING * 2), w - (PADDING * 2)
    (file_names, inputs) = load_images(input_dir, (org_h, org_w, c))
    inputs = np.pad(inputs, [(0, 0), (PADDING, PADDING), (PADDING, PADDING), (0, 0)], 'constant', constant_values=0)

    network = UNet(INPUT_IMAGE_SHAPE, CLASS_NUM)

    model = network.model()
    plot_model(model, to_file='../model_plot.png')
    model.summary()
    if isinstance(gpu_num, int):
        model = multi_gpu_model(model, gpus=gpu_num)
    model.load_weights(os.path.join(DIR_MODEL, File_MODEL))
    print('predicting ...')
    preds = model.predict(inputs, BATCH_SIZE)
    print('... predicted')

    print('output saveing ...')
    preds = preds[:, PADDING:org_h+PADDING, PADDING:org_w+PADDING, :]
    save_images(DIR_OUTPUTS, preds, file_names)
    print('... saved')


if __name__ == '__main__':
    args = get_option(EPOCHS)
    EPOCHS = args.epoch

    if not(os.path.exists(DIR_MODEL)):
        os.mkdir(DIR_MODEL)
    if not(os.path.exists(DIR_OUTPUTS)):
        os.mkdir(DIR_OUTPUTS)

    train(gpu_num=GPU_NUM, with_generator=False, load_model=False)
    #train(gpu_num=GPU_NUM, with_generator=True, load_model=False)

    #predict(DIR_INPUTS, gpu_num=GPU_NUM)
    #predict(DIR_PREDICTS, gpu_num=GPU_NUM)
