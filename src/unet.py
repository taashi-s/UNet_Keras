from keras.models import Model
from keras.engine.topology import Input
from keras.layers.core import Flatten, Dense, Reshape, Dropout, Activation
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization


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

        add_drop_layer_indexes = [2, 3]
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

    def model(self):
        return self.MODEL
