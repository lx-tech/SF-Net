 # -*- coding: utf-8 -*-
import numpy as np
import cv2
#import JackFramework as jf


class DataSaver(object):
    """docstring for DataSaver"""

    def __init__(self, args: object) -> object:
        super().__init__()
        self.__args = args

    def save_output(filename, output):
        image = np.squeeze(output[0].detach().cpu().numpy())
        cv2.imwrite(filename, image.astype(np.uint16))


    def save_output_color(filename, output):
        image = output[0].permute(1, 2, 0).detach().cpu().numpy()
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, image)
