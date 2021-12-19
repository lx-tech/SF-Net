# -*- coding: utf-8 -*-
import time
import JackFramework as jf
# import UserModelImplementation.user_define as user_def
from .data_set import BodyReconstructionDataset
#import data_saver

# traing
ID_COLOR = 0
ID_DEPTH = 1
ID_COLOR_GT =2
ID_DEPTH_GT =3
#testing

class BodyDataloader(jf.UserTemplate.DataHandlerTemplate):
    """docstring for DataHandlerTemplate"""

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args
        self.__result_str = jf.ResultStr()
        self.__train_dataset = None
        self.__val_dataset = None
        #self.__saver = StereoSaver(args)
        self.__start_time = 0
        

    def get_train_dataset(self, path: str, is_training: bool = True) -> object:
        args = self.__args
        self.__train_dataset = BodyReconstructionDataset(args, args.trainListPath, is_training)
        return self.__train_dataset

    def get_val_dataset(self, path: str) -> object:
        # return dataset
        args = self.__args
        self.__val_dataset = BodyReconstructionDataset(args, args.valListPath, False)
        return self.__val_dataset

    def split_data(self, batch_data: tuple, is_training: bool) -> list:
        self.__start_time = time.time()
        if is_training:
            # return input_data_list, label_data_list
             return [batch_data[ID_COLOR], batch_data[ID_DEPTH]], [batch_data[ID_COLOR_GT], batch_data[ID_DEPTH_GT]]
            # return input_data, supplement
        return [batch_data[ID_COLOR], batch_data[ID_DEPTH]]

    def show_train_result(self, epoch: int, loss:
                          list, acc: list,
                          duration: float) -> None:
        assert len(loss) == len(acc)  # same model number
        info_str = self.__result_str.training_result_str(epoch, loss[0], acc[0], duration, True)
        jf.log.info(info_str)

    def show_val_result(self, epoch: int, loss:
                        list, acc: list,
                        duration: float) -> None:
        assert len(loss) == len(acc)  # same model number
        info_str = self.__result_str.training_result_str(epoch, loss[0], acc[0], duration, False)
        jf.log.info(info_str)

    def save_result(self, output_data: list, supplement: list,
                    img_id: int, model_id: int) -> None:
        assert self.__train_dataset is not None
        args = self.__args
        # save method       
        pass

    

    def show_intermediate_result(self, epoch: int,
                                 loss: list, acc: list) -> str:
        assert len(loss) == len(acc)  # same model number
        return self.__result_str.training_intermediate_result(epoch, loss[0], acc[0])