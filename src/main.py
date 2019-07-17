import os
import time
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


CLASS_NUM = 1
PADDING = 0
INPUT_IMAGE_SHAPE = (256 + (PADDING * 2), 256 + (PADDING * 2), 3)
BATCH_SIZE = 250
EPOCHS = 600
GPU_NUM = 8


SUFIX = ''
DIR_BASE = os.path.join('.', '..')
DIR_MODEL = os.path.join(DIR_BASE, 'model')
DIR_TRAIN_INPUTS = os.path.join(DIR_BASE, 'Train_inputs' + SUFIX)
DIR_TRAIN_TEACHERS = os.path.join(DIR_BASE, 'Train_teachers' + SUFIX)
DIR_VALID_INPUTS = os.path.join(DIR_BASE, 'Valid_inputs' + SUFIX)
DIR_VALID_TEACHERS = os.path.join(DIR_BASE, 'Valid_teachers' + SUFIX)
DIR_OUTPUTS = os.path.join(DIR_BASE, 'outputs' + SUFIX)
DIR_TEST = os.path.join(DIR_BASE, 'predict_data')
DIR_PREDICTS = os.path.join(DIR_BASE, 'predict_data' + SUFIX)

DIR_LEARN_INPUTS = os.path.join(DIR_BASE, 'learn_inputs' + SUFIX)
DIR_LEARN_OUTPUTS = os.path.join(DIR_BASE, 'learn_outputs' + SUFIX)


File_MODEL = 'Model_2019_0716_1620_Deep_layers_train_only_weights.hdf5'

#FILTER_LIST = [32, 64, 128, 256, 512]
#FILTER_LIST = [32, 64, 128]
#FILTER_LIST = [32, 64, 128, 256]
FILTER_LIST = [32, 64, 128, 256, 512]
TRAINABLE_LIST = [False, False, True, True, True]

def train(gpu_num=None, with_generator=False, load_model=False, show_info=True):
    print('network creating ... ')#, end='', flush=True)
    network = UNet(INPUT_IMAGE_SHAPE, CLASS_NUM, filters_list=FILTER_LIST, trainable_list=TRAINABLE_LIST)
    print('... created')

    model = network.model()
    if show_info:
        model.summary()
    if isinstance(gpu_num, int):
        model = multi_gpu_model(model, gpus=gpu_num)

    model_filename = os.path.join(DIR_MODEL, File_MODEL)
    model_filename_best = os.path.join(DIR_MODEL, File_MODEL[:-5] + '_best.hdf5')
    if load_model:
        print('loading weghts ... ', end='', flush=True)
        model.load_weights(model_filename)
        print('... loaded')

    model.compile(optimizer=Adam(lr=0.001), loss=DiceLossByClass(INPUT_IMAGE_SHAPE, CLASS_NUM).dice_coef_loss)

    callbacks = [ KC.TensorBoard()
                , HistoryCheckpoint(filepath='LearningCurve_{history}.png'
                                    , verbose=1
                                    , period=2
                                   )
                , KC.ModelCheckpoint(filepath=model_filename_best
                                     , verbose=1
                                     , save_weights_only=True
                                     , save_best_only=True
                                     , period=2
                                    )
                , KC.ModelCheckpoint(filepath=model_filename
                                     , verbose=1
                                     , save_weights_only=True
                                     , save_best_only=False
                                     , period=2
                                    )
                ]

    print('data generating ...', end='', flush=True)
    train_generator = DataGenerator(DIR_TRAIN_INPUTS, DIR_TRAIN_TEACHERS, INPUT_IMAGE_SHAPE, CLASS_NUM
                                    , include_padding=(PADDING, PADDING))
    valid_generator = DataGenerator(DIR_VALID_INPUTS, DIR_VALID_TEACHERS, INPUT_IMAGE_SHAPE, CLASS_NUM
                                    , include_padding=(PADDING, PADDING))
    print('... created')

    if with_generator:
        train_data_num = train_generator.data_size()
        valid_data_num = valid_generator.data_size()
        ### tmp : supress step >>>
        #supress_step = 100
        #print('supress step : ', math.ceil(train_data_num / BATCH_SIZE), ' --> ', supress_step)
        ### tmp : supress step <<<
        history = model.fit_generator(train_generator.generator(batch_size=BATCH_SIZE)
                                      #, steps_per_epoch=supress_step#math.ceil(train_data_num / BATCH_SIZE)
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
        """
        ccc = BATCH_SIZE * 10
        print('@@@@@ use ', ccc, ' data')
        print('&&&&& ', np.shape(inputs))
        train_tuple = ([inputs[0][:ccc]], [teachers[0][:ccc]])#list(zip(inputs[:ccc], teachers[:ccc]))
        random.shuffle(train_tuple)
        inputs, teachers = train_tuple
        valid_data = valid_data[:ccc]
        print('@@@@@ use i ', np.shape(inputs), ' data')
        print('@@@@@ use t ', np.shape(teachers), ' data')
        file_names = ['chk_%06d.png' % k for k in range(len(inputs[0]))]
        save_images(os.path.join(DIR_BASE, 'chk_data', 'is'), inputs[0], file_names)
        save_images(os.path.join(DIR_BASE, 'chk_data', 'ts'), teachers[0], file_names)
        #"""
        print('... generated')
        history = model.fit(inputs, teachers, batch_size=BATCH_SIZE, epochs=EPOCHS
                            , validation_data=valid_data
                            , shuffle=True, verbose=1, callbacks=callbacks)

    print('model saveing ... ', end='', flush=True)
    model.save_weights(model_filename)
    print('... saved')
    print('learning_curve saveing ... ', end='', flush=True)
    #save_learning_curve(history)
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


def predict(input_dir, gpu_num=None, out_dir=None, is_suppress=False):
    start_t = time.time()
    h, w, c = INPUT_IMAGE_SHAPE
    org_h, org_w = h - (PADDING * 2), w - (PADDING * 2)
    (file_names, inputs) = load_images(input_dir, (org_h, org_w, c))
    inputs = np.pad(inputs, [(0, 0), (PADDING, PADDING), (PADDING, PADDING), (0, 0)], 'constant', constant_values=0)

    if is_suppress:
        ###### tmp : data suppress
        bfo_suppress_count = len(file_names)
        data_suppress_num = 20
        file_names = file_names[:data_suppress_num] + file_names[-data_suppress_num:]
        inputs = inputs[:data_suppress_num] + inputs[-data_suppress_num:]
        print('aft suppress : ', bfo_suppress_count, ' --> ', len(file_names))

    network = UNet(INPUT_IMAGE_SHAPE, CLASS_NUM, filters_list=FILTER_LIST, trainable_list=TRAINABLE_LIST)
    create_net_t = time.time()

    model = network.model()
#    plot_model(model, to_file='../model_plot.png')
#    model.summary()
    if isinstance(gpu_num, int):
        model = multi_gpu_model(model, gpus=gpu_num)
    load_start_t = time.time()
    #model.load_weights(os.path.join(DIR_MODEL, File_MODEL))
    model.load_weights(os.path.join(DIR_MODEL, File_MODEL[:-5] + '_best.hdf5'))
    load_finish_t = time.time()

    make_p_start_t = time.time()
    model._make_predict_function()
    make_p_finish_t = time.time()

    predict_times = []
    save_times = []

    if out_dir is None:
        out_dir = DIR_OUTPUTS

    target_zip = zip(inputs, file_names)
    for inp, fname in target_zip:
        pred_start_t = time.time()
        inp = np.array([inp])
        preds = model.predict(inp, BATCH_SIZE)
        pred_finish_t = time.time()
 
        save_start_t = time.time()
        preds = preds[:, PADDING:org_h+PADDING, PADDING:org_w+PADDING, :]
        save_images(out_dir, preds, [fname])
        save_finish_t = time.time()
        predict_times.append(pred_finish_t - pred_start_t)
        save_times.append(save_finish_t- save_start_t)

    """
    print('predicting ...')
    pred_start_t = time.time()
    preds = model.predict(inputs, BATCH_SIZE)
    pred_finish_t = time.time()
    print('... predicted')

    print('output saveing ...')
    save_start_t = time.time()
    preds = preds[:, PADDING:org_h+PADDING, PADDING:org_w+PADDING, :]
    save_images(DIR_OUTPUTS, preds, file_names)
    save_finish_t = time.time()
    print('... saved')
    """

    print('##### Predict Times (average) [target files : ', len(file_names), ']')
    time_create_net = create_net_t - start_t
    time_load_weights = load_finish_t - load_start_t
    time_make_p = make_p_finish_t - make_p_start_t
    time_1st_p = predict_times[0]
    time_ave_p = np.average(np.array(predict_times[1:]))
    time_min_p = min(predict_times[1:])
    time_max_p = max(predict_times[1:])
    time_ave_s = np.average(np.array(save_times))
    time_min_s = min(save_times)
    time_max_s = max(save_times)
    time_total = save_finish_t - start_t

    print('create_net : ', time_create_net)
    print('model.load_weights : ', time_load_weights)
    print('model._make_predict_function : ', time_make_p)
    print('model.predict 1st call: ', time_1st_p)
    print('model.predict average: ', time_ave_p, ' [', time_min_p, ' - ', time_max_p, ']')
    print('save_images average : ', time_ave_s, ' [', time_min_s, ' - ', time_max_s, ']')
    print('total : ', time_total)
    return time_create_net, time_load_weights, time_make_p, time_1st_p, time_ave_p, time_min_p, time_max_p, time_ave_s, time_min_s, time_max_s, time_total


if __name__ == '__main__':
    args = get_option(EPOCHS)
    EPOCHS = args.epoch

    if not(os.path.exists(DIR_MODEL)):
        os.mkdir(DIR_MODEL)
    if not(os.path.exists(DIR_OUTPUTS)):
        os.mkdir(DIR_OUTPUTS)

    train(gpu_num=GPU_NUM, with_generator=False, load_model=False)
    #train(gpu_num=GPU_NUM, with_generator=True, load_model=False)

    predict(DIR_LEARN_INPUTS, gpu_num=GPU_NUM, out_dir=DIR_LEARN_OUTPUTS)
    predict(DIR_PREDICTS, gpu_num=GPU_NUM)

