import keras.backend as KB
import tensorflow as tf

class DiceLossByClass():
    def __init__(self):
        pass

    def dice_coef(self, y_true, y_pred):
        y_true = KB.flatten(y_true)
        y_pred = KB.flatten(y_pred)
        intersection = KB.sum(y_true * y_pred)
        return (2.0 * intersection + 1) / (KB.sum(y_true) + KB.sum(y_pred) + 1)


    def dice_coef_loss(self, y_true, y_pred):
        # (N, h, w, ch)
        y_trues = tf.unstack(y_true, axis=3)
        y_preds = tf.unstack(y_pred, axis=3)

        losses = 0
        for y_t, y_p in zip(y_trues, y_preds):
            losses += self.dice_coef(y_t, y_p)

        return -1 * losses
