import keras.backend as Kbackend


def dice_coef(y_true, y_pred):
    y_true = Kbackend.flatten(y_true)
    y_pred = Kbackend.flatten(y_pred)
    intersection = Kbackend.sum(y_true * y_pred)
    return (2.0 * intersection + 1) / (Kbackend.sum(y_true) + Kbackend.sum(y_pred) + 1)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
