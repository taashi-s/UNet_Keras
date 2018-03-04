from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, UpSampling2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate


class UNet(object):
    def __init__(self):

        inputs = Input((572, 572, 1))

        encodeLayer1 = self.__add_Encode_layers(64, inputs, is_first=True)
        encodeLayer2 = self.__add_Encode_layers(128, encodeLayer1)
        encodeLayer3 = self.__add_Encode_layers(256, encodeLayer2)
        encodeLayer4 = self.__add_Encode_layers(512, encodeLayer3)
        encodeLayer5 = self.__add_Encode_layers(1024, encodeLayer4)

        decodeLayer1 = self.__add_Decode_layers(
            512, encodeLayer5, encodeLayer4)
        decodeLayer2 = self.__add_Decode_layers(
            256, decodeLayer1, encodeLayer3)
        decodeLayer3 = self.__add_Decode_layers(
            128, decodeLayer2, encodeLayer2)
        decodeLayer4 = self.__add_Decode_layers(64, decodeLayer3, encodeLayer1)

        outputs = Conv2D(1, 1, activation='relu')(decodeLayer4)

        self.MODEL = Model(inputs=inputs, outputs=outputs)

    def __add_Encode_layers(self, filters, inputLayer, is_first=False):
        layer = inputLayer
        if is_first:
            layer = Conv2D(filters, 3, activation='relu',
                           input_shape=(572, 572, 1))(layer)
        else:
            layer = MaxPooling2D(2)(layer)
            layer = Conv2D(filters, 3, activation='relu')(layer)
        layer = Conv2D(filters, 3, activation='relu')(layer)
        return layer

    def __add_Decode_layers(self, filters, inputLayer, concatLayer):
        layer = UpSampling2D(2)(inputLayer)
        layerShape = layer.get_shape().as_list()
        concatLayerShape = concatLayer.get_shape().as_list()
        crop_w = (concatLayerShape[1] - layerShape[1]) // 2
        crop_h = (concatLayerShape[2] - layerShape[2]) // 2
        concatLayer = Cropping2D((crop_w, crop_h))(concatLayer)

        layer = concatenate([layer, concatLayer], axis=3)
        layer = Conv2D(filters, 3, activation='relu')(layer)
        layer = Conv2D(filters, 3, activation='relu')(layer)
        layer = MaxPooling2D(2)(layer)
        return layer

    def model(self):
        return self.MODEL
