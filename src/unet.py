import math
from keras.models import Model
from keras.engine.topology import Input
from keras.layers.core import Flatten, Dense, Reshape, Dropout, Activation
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers import ZeroPadding2D


class UNet(object):
    def __init__(self, input_shape, class_num, filters_list=None):
        self.INPUT_SHAPE = input_shape
        self.CLASS_NUM = class_num

        inputs = Input(self.INPUT_SHAPE)

        if filters_list is None:
            filters_list = [32, 64, 128, 256, 512]
        layer = inputs
        encodeLayers = []
        for k, filters in enumerate(filters_list):
            layer = self.__add_encode_layers(filters, layer, is_first=(k==0))
            encodeLayers.append(layer)

        add_drop_layer_indexes = [1, 2, 3]
        deconv_item = zip(reversed(filters_list[:-1]), reversed(encodeLayers[:-1]))
        for k, (filters, concat_layer) in enumerate(deconv_item):
            layer = self.__add_decode_layers(filters, layer, concat_layer
                                             , add_drop_layer=(k in add_drop_layer_indexes))

        outputs = Conv2D(class_num, 1, activation='sigmoid')(layer)

        self.MODEL = Model(inputs=[inputs], outputs=[outputs])

    def __add_encode_layers(self, filter_size, input_layer, is_first=False):
        layer = input_layer
        if is_first:
            layer = Conv2D(filter_size, 3, padding='same', input_shape=self.INPUT_SHAPE)(layer)
        else:
            layer = MaxPooling2D(2)(layer)
            layer = Conv2D(filter_size, 3, padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation(activation='relu')(layer)

        layer = Conv2D(filter_size, 3, padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation(activation='relu')(layer)
        return layer

    def __add_decode_layers(self, filter_size, input_layer, concat_layer, add_drop_layer=False):
        layer = UpSampling2D(2)(input_layer)
        layer, concat_layer = self.__adjustment_shape(layer, concat_layer)
        layer = concatenate([layer, concat_layer])

        layer = Conv2D(filter_size, 3, padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation(activation='relu')(layer)

        layer = Conv2D(filter_size, 3, padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation(activation='relu')(layer)

        if add_drop_layer:
            layer = Dropout(0.5)(layer)
        return layer


    def __adjustment_shape(self, input_layer, concat_layer):
        _, h_i, w_i, _ = input_layer.get_shape().as_list()
        _, h_c, w_c, _ = concat_layer.get_shape().as_list()

        if h_i < h_c:
            #crop_hs = self.__get_crop_size(h_i, h_c)
            #concat_layer = Cropping2D(cropping=(crop_hs, (0, 0)))(concat_layer)
            pad_hs = self.__get_crop_size(h_i, h_c)
            input_layer = ZeroPadding2D(padding=(pad_hs, (0, 0)))(input_layer)
        elif h_c < h_i:
            #crop_hs = self.__get_crop_size(h_i, h_c)
            #input_layer = Cropping2D(cropping=(crop_hs, (0, 0)))(input_layer)
            pad_hs = self.__get_crop_size(h_i, h_c)
            concat_layer = ZeroPadding2D(padding=(pad_hs, (0, 0)))(concat_layer)

        if w_i < w_c:
            #crop_ws = self.__get_crop_size(w_i, w_c)
            #concat_layer = Cropping2D(cropping=((0, 0), crop_ws))(concat_layer)
            pad_ws = self.__get_crop_size(w_i, w_c)
            input_layer = ZeroPadding2D(padding=((0, 0), pad_ws))(input_layer)
        elif w_c < w_i:
            #crop_ws = self.__get_crop_size(w_i, w_c)
            #input_layer = Cropping2D(cropping=((0, 0), crop_ws))(input_layer)
            pad_ws = self.__get_crop_size(w_i, w_c)
            concat_layer = ZeroPadding2D(padding=((0, 0), pad_ws))(concat_layer)

        return input_layer, concat_layer


    def __get_crop_size(self, target, crop):
        pad = (crop - target) / 2
        crop_size = (int(pad), int(pad))
        if pad != int(pad):
            crop_size = (math.ceil(pad), int(pad))
        return crop_size


    def model(self):
        return self.MODEL
