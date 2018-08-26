import os
import glob
import cv2
import random
import numpy as np
import time

import images_loader as iml


class DataGenerator():
    def __init__(self, input_dir, teacher_dir, image_shape, include_padding=None):
      self.__input_dir = input_dir
      self.__teacher_dir = teacher_dir
      self.__padding = (0, 0)
      if isinstance(include_padding, tuple):
          self.__padding = include_padding

      h, w, c = image_shape
      h_pad, w_pad = self.__padding
      self.__image_shape = (h - (h_pad * 2), w - (w_pad * 2), c)
      
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


    def generate_data(self, target_data_list=None):
        data_list = self.__data_names
        if target_data_list is not None:
            data_list = target_data_list

        data_num = len(data_list)
        start = time.time()
        prev_time = start

        input_list = []
        teacher_list = []
        random.shuffle(data_list)
        for k, name in enumerate(data_list):
            input_img, teacher_img = self.load_data(name)
            if input_img is None or teacher_img is None:
                continue

            input_list.append(input_img)
            teacher_list.append(teacher_img)
            if k % 200 == 0 or k == data_num - 1:
                now_time = time.time()
                print('## generate : ', '%05d' % k, '/', '%05d' % data_num, ' %5.3f(%5.3f)' % (now_time - prev_time, now_time - start))
                prev_time = now_time

        inputs = [np.array(input_list)]
        teachers = [np.array(teacher_list)]

        #print('inputs : ', np.shape(inputs), ', teachers : ', np.shape(teachers))

        return inputs, teachers


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

                input_img, teacher_img = self.load_data(name)
                if input_img is None or teacher_img is None:
                    continue

                input_list.append(input_img)
                teacher_list.append(teacher_img)

                if len(input_list) >= batch_size:
                    inputs = [np.array(input_list)]
                    teachers = [np.array(teacher_list)]

                    yield inputs, teachers

    def load_data(self, name):
        input_path = os.path.join(self.__input_dir, name)
        teacher_path = os.path.join(self.__teacher_dir, name)
        input_img = iml.load_image(input_path, self.__image_shape, with_normalize=True)
        #teacher_img = iml.load_image(teacher_path, self.__image_shape, with_normalize=True)
        teacher_shape = self.__image_shape # (self.__image_shape[0], self.__image_shape[1], 1)
        teacher_img = iml.load_image(teacher_path, teacher_shape, with_normalize=True)
        #h, w, _ = teacher_shape
        #teacher_img[0, :, :] = 1
        #teacher_img[h - 1, :, :] = 1
        #teacher_img[:, 0, :] = 1
        #teacher_img[:, w - 1, :] = 1
#        input_img = self.padding_data(input_img, 0)
#        teacher_img = self.padding_data(teacher_img, 1)
        return input_img, teacher_img


    def padding_data(self, data, padding_val):
        h_pad, w_pad = self.__padding
        return np.pad(data, [(h_pad, h_pad), (w_pad, w_pad), (0, 0)], 'constant', constant_values=padding_val)
