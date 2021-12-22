 # -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
#import JackFramework as jf


class DataSaver(object):
    """docstring for DataSaver"""
    _DEPTH_UNIT = 1000.0
    _DEPTH_DIVIDING = 255.0

    def __init__(self, args: object) -> object:
        super().__init__()
        self.__args = args

    def save_output(self, color_pre: np.array,  
                    img_id: int, dataset_name: str,
                    supplement: list) -> None:
        #print(color_pre.shape)
        batch_size, _, _, _, = color_pre.shape
        #names = supplement[0]
        for i in range(batch_size):
            temp_color = color_pre[i,:,:,:]
            #temp_depth = depth_pre[i,:,:,:]

            name = batch_size * img_id + i

            self._save_output_color(temp_color, name)
            #self._save_output_depth(temp_depth, name)

    def _save_output_depth(self, img: np.array, num: int) -> None:
        args = self.__args
        path = self._generate_output_img_path(args.resultImgDir, num, "%04d_depth_front")        
        img =  img.transpose(1, 2, 0)
        img = (img * float(DataSaver._DEPTH_UNIT)).astype(np.uint16)
        print(img)
        cv2.imwrite(path, img)


    def _save_output_color(self, img: np.array, num: int) -> None:
        args = self.__args
        path = self._generate_output_img_path(args.resultImgDir, num, "%04d_color_front")
        #img = np.squeeze(img)
        img = img.transpose(1, 2, 0)
        #img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        #cv2.imwrite(path, img)
        img = (img  * float(DataSaver._DEPTH_DIVIDING)).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img)

    @staticmethod
    def _generate_output_img_path(dir_path: str, num: str,
                                  filename_format: str = "%04d_10",
                                  img_type: str = ".png"):
        return dir_path + filename_format % num + img_type