from keras.models import Model
from keras.layers import Input, Activation, Dropout, LeakyReLU, BatchNormalization
from keras.layers.core import Flatten, Dense, Reshape
from keras.layers.convolutional import Conv2D, UpSampling2D, Cropping2D, Conv2DTranspose, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate


class UNet_n(object):
    def __init__(self, input_size):
        self.INPUT_SIZE = input_size

        inputs = Input((self.INPUT_SIZE, self.INPUT_SIZE, 1))
        print(inputs.shape)

        encodeLayer0 = ZeroPadding2D((1, 1))(inputs)
        encodeLayer1 = Conv2D(64, 4, strides=2)(encodeLayer0)
        encodeLayer2 = self.__add_Encode_layers(128, encodeLayer1)
        encodeLayer3 = self.__add_Encode_layers(256, encodeLayer2)
        encodeLayer4 = self.__add_Encode_layers(512, encodeLayer3)
        encodeLayer5 = self.__add_Encode_layers(512, encodeLayer4)

        decodeLayer0 = self.__add_Decode_layers(
            512, encodeLayer5, encodeLayer4)
        decodeLayer1 = self.__add_Decode_layers(
            256, decodeLayer0, encodeLayer3)
        decodeLayer2 = self.__add_Decode_layers(
            128, decodeLayer1, encodeLayer2, addDropLayer=True)
        decodeLayer3 = self.__add_Decode_layers(
            64, decodeLayer2, encodeLayer1, addDropLayer=True)

        outputs = Activation(activation='relu')(decodeLayer3)
        outputs = Conv2DTranspose(1, 2, strides=2)(outputs)
        outputs = Activation(activation='sigmoid')(outputs)

        print(outputs.shape)

        self.MODEL = Model(inputs=[inputs], outputs=[outputs])

    def __add_Encode_layers(self, filters, inputLayer):
        layer = LeakyReLU(0.2)(inputLayer)
        layer = ZeroPadding2D((1, 1))(layer)
        layer = Conv2D(filters, 4, strides=2)(layer)
        layer = BatchNormalization()(layer)
        print(layer.shape)
        return layer

    def __add_Decode_layers(self, filters, inputLayer, concatLayer, addDropLayer=False):
        layer = Activation(activation='relu')(inputLayer)
        layer = Conv2DTranspose(filters, 2, strides=2,
                                kernel_initializer='he_uniform')(layer)
        layer = BatchNormalization()(layer)
        if addDropLayer:
            layer = Dropout(0.5)(layer)
        layer = concatenate([layer, concatLayer], axis=-1)
        print(layer.shape)
        return layer

    def model(self):
        return self.MODEL
