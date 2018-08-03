import os
import glob
import cv2
import random
import numpy as np

import images_loader as iml


class DataGenerator():
    def __init__(self, input_dir, teacher_dir, image_shape):
      self.__input_dir = input_dir
      self.__teacher_dir = teacher_dir
      self.__image_shape = image_shape
      self.__update_data_names()


    def __update_data_names(self):
        files = glob.glob(os.path.join(self.__input_dir, '*.png'))
        files += glob.glob(os.path.join(self.__input_dir, '*.jpeg'))
        files += glob.glob(os.path.join(self.__input_dir, '*.jpg'))
        files.sort()
        self.__data_names = []
        for file in files:
          # TODO : Support other extension
          name = os.path.basename(file)
          teacher_path = os.path.join(self.__teacher_dir, name)
          if not os.path.exists(teacher_path):
              continue
          self.__data_names.append(name)


    def data_size(self):
        return len(self.__data_names)


    def generator(self, batch_size=None, target_data_list=None):
        """
        keras data generator
        """
        data_list = self.__data_names
        if target_data_list is not None:
            data_list = target_data_list

        if batch_size is None:
            batch_size = self.data_size()

        input_list = []

        while True:
            random.shuffle(data_list)
            for name in data_list:
                if (input_list == []) or (len(input_list) >= batch_size):
                    input_list = []
                    teacher_list = []

                input_path = os.path.join(self.__input_dir, name)
                teacher_path = os.path.join(self.__teacher_dir, name)
                input_img = iml.load_image(input_path, self.__image_shape, with_normalize=True)
                teacher_img = iml.load_image(teacher_path, self.__image_shape, with_normalize=True)

                if input_img is None or teacher_img is None:
                    continue

                input_list.append(input_img)
                teacher_list.append(teacher_img)

                if len(input_list) >= batch_size:
                    inputs = [np.array(input_list)]
                    teachers = [np.array(teacher_list)]

                    # print('')
                    # for k, inp in enumerate(inputs):
                    #    print('input(', k, ')>>> ', np.shape(inp))

                    yield inputs, teachers

