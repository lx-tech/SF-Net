# -*- coding: utf-8 -*-
import torch
# import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim

import JackFramework as jf
import sys
# import UserModelImplementation.user_define as user_def
from .model import Model


class BodyReconstructionInterface(jf.UserTemplate.ModelHandlerTemplate):
    """docstring for BodyReconstructionInterface"""

    MODEL_ID = 0  # only color_net
    COLOR_LABEL_ID = 0  # with 2 label
    DEPTH_LABLE_ID = 1
    COLOR_ID = 0
    DEPTH_ID = 1

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args

    @staticmethod
    def lr_lambda(epoch: int) -> float:
        max_warm_up_epoch = 10
        convert_epoch = 50
        off_set = 1
        lr_factor = 1.0

        factor = ((epoch + off_set) / max_warm_up_epoch) if epoch < max_warm_up_epoch \
            else lr_factor if (epoch >= max_warm_up_epoch and epoch < convert_epoch) \
            else lr_factor * 0.25
        return factor

    def get_model(self) -> list:
        args = self.__args
        self.__lr = args.lr
        # return model
        ngf = 32
        model = Model(in_channel=4, out_channel=3, ngf=ngf, upconv=False, norm=True)
        return [model]

    def optimizer(self, model: list, lr: float) -> list:
        args = self.__args

        opt = optim.Adam(model[0].parameters(), lr=lr)

        if args.lr_scheduler:
            sch = optim.lr_scheduler.LambdaLR(opt, lr_lambda=self.lr_lambda)
        else:
            sch = None

        return [opt], [sch]
        

    def lr_scheduler(self, sch: object, ave_loss: list, sch_id: int) -> None:
        # how to do schenduler
        if self.MODEL_ID == sch_id:
            sch.step()

    def inference(self, model: list, input_data: list, model_id: int) -> list:
        # args = self.__args
        if self.MODEL_ID == model_id:
            color_front, depth_front = model(torch.cat((input_data[self.COLOR_ID], input_data[self.DEPTH_ID]), dim=1), 
                                            torch.cat((input_data[self.COLOR_ID], input_data[self.DEPTH_ID]), dim=1))
        return [color_front, depth_front]


    def accuary(self, output_data: list, label_data: list, model_id: int) -> list:
        # return acc
        # args = self.__args
        acc_0 = None
        acc_1 = None
        if self.MODEL_ID == model_id:
            acc_0 = jf.BaseAccuracy.rmse_score(output_data[0], label_data[0])
            acc_1 = jf.BaseAccuracy.rmse_score(output_data[1], label_data[1])
        return [acc_0, acc_1]

    def loss(self, output_data: list, label_data: list, model_id: int) -> list:
        # return loss
        # args = self.__args
        total_loss = None
        loss_0 = None
        loss_1 = None
        if self.MODEL_ID == model_id:
            loss_0 = jf.Loss.focal_loss(output_data[0], label_data[0])
            loss_1 = jf.Loss.focal_loss(output_data[1], label_data[1])
            total_loss = loss_0 + loss_1

        return [total_loss, loss_0, loss_1]

    # Optional
    def pretreatment(self, epoch: int, rank: object) -> None:
        # do something before training epoch
        pass

    # Optional
    def postprocess(self, epoch: int, rank: object,
                    ave_tower_loss: list, ave_tower_acc: list) -> None:
        # do something after training epoch
        pass

    # Optional
    def load_model(self, model: object, checkpoint: dict, model_id: int) -> bool:
        # return False
        return False

    # Optional
    def load_opt(self, opt: object, checkpoint: dict, model_id: int) -> bool:
        # return False
        return False

    # Optional
    def save_model(self, epoch: int, model_list: list, opt_list: list) -> dict:
        # return None
        return None
