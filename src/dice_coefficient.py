import keras.backend as KB
import tensorflow as tf
import numpy as np

class DiceLossByClass():
    def __init__(self, input_shape, class_num):
        self.__input_h = input_shape[0]
        self.__input_w = input_shape[1]
        self.__class_num = class_num
        

    def dice_coef(self, y_true, y_pred):
        y_true = KB.flatten(y_true)
        y_pred = KB.flatten(y_pred)
        intersection = KB.sum(y_true * y_pred)
        return (2.0 * intersection + 1) / (KB.sum(y_true) + KB.sum(y_pred) + 1)


    def dice_coef_loss(self, y_true, y_pred):
        # (N, h, w, ch)
        y_true_res = tf.reshape(y_true, (-1, self.__input_h, self.__input_w, self.__class_num))
        y_pred_res = tf.reshape(y_pred, (-1, self.__input_h, self.__input_w, self.__class_num))
        y_trues = tf.unstack(y_true_res, axis=3)
        y_preds = tf.unstack(y_pred_res, axis=3)

        losses = []
        for y_t, y_p in zip(y_trues, y_preds):
            losses.append(1 - self.dice_coef(y_t, y_p))

        return tf.reduce_mean(tf.stack(losses))
