from keras.models import Model
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Reshape
from keras.layers.convolutional import Conv2D, UpSampling2D, Cropping2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate


class UNet(object):
    def __init__(self, input_size):
        self.INPUT_SIZE = input_size

        inputs = Input((self.INPUT_SIZE, self.INPUT_SIZE, 1))

        encodeLayer1 = self.__add_Encode_layers(64, inputs, is_first=True)
        encodeLayer2 = self.__add_Encode_layers(128, encodeLayer1)
        encodeLayer3 = self.__add_Encode_layers(256, encodeLayer2)
        encodeLayer4 = self.__add_Encode_layers(512, encodeLayer3)
        # encodeLayer5 = self.__add_Encode_layers(1024, encodeLayer4)
        encodeLayer5 = encodeLayer4

        # decodeLayer1 = self.__add_Decode_layers(
        #     512, encodeLayer5, encodeLayer4)
        decodeLayer1 = encodeLayer5
        print(decodeLayer1.shape)
        decodeLayer2 = self.__add_Decode_layers(
            256, decodeLayer1, encodeLayer3)
        print(decodeLayer2.shape)
        decodeLayer3 = self.__add_Decode_layers(
            128, decodeLayer2, encodeLayer2)
        print(decodeLayer3.shape)
        decodeLayer4 = self.__add_Decode_layers(64, decodeLayer3, encodeLayer1)
        print(decodeLayer4.shape)

        outputs = Conv2D(1, 1, activation='sigmoid')(decodeLayer4)

        outputLayerShape = outputs.get_shape().as_list()
        if (outputLayerShape[1], outputLayerShape[2]) != (self.INPUT_SIZE, self.INPUT_SIZE):
            outputs = Flatten()(outputs)
            outputs = Dense(self.INPUT_SIZE*self.INPUT_SIZE*1)(outputs)
            outputs = Reshape((self.INPUT_SIZE, self.INPUT_SIZE, 1))(outputs)

        print(outputs.shape)

        self.MODEL = Model(inputs=[inputs], outputs=[outputs])

    def __add_Encode_layers(self, filters, inputLayer, is_first=False):
        layer = inputLayer
        if is_first:
            layer = Conv2D(filters, 3, activation='relu',
                           input_shape=(self.INPUT_SIZE, self.INPUT_SIZE, 1))(layer)
        else:
            layer = MaxPooling2D(2)(layer)
            layer = Conv2D(filters, 3, activation='relu')(layer)
        layer = Conv2D(filters, 3, activation='relu')(layer)
        return layer

    def __add_Decode_layers(self, filters, inputLayer, concatLayer):
        layer = UpSampling2D(2)(inputLayer)
        layerShape = layer.get_shape().as_list()
        concatLayerShape = concatLayer.get_shape().as_list()
        diff_w = concatLayerShape[1] - layerShape[1]
        diff_h = concatLayerShape[2] - layerShape[2]
        crop_l = diff_w // 2
        crop_t = diff_h // 2
        crop_r = crop_l
        if crop_l != (diff_w / 2):
            crop_r = crop_r + 1
        crop_b = crop_t
        if crop_t != (diff_h / 2):
            crop_b = crop_b + 1
        concatLayer = Cropping2D(
            ((crop_t, crop_b), (crop_l, crop_r)))(concatLayer)

        layer = concatenate([layer, concatLayer])
        layer = Conv2D(filters, 3, activation='relu')(layer)
        layer = Conv2D(filters, 3, activation='relu')(layer)
        return layer

    def model(self):
        return self.MODEL
