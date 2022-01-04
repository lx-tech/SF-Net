import torch
import torch.nn as nn
#import sys
#sys.path.append("./Source/UserModelImplementation/Models/BodyReconstruction")
from .submodel import UNet, SUNet

class Model(nn.Module):
    """docstring for Model"""

    def __init__(self, in_channel: int, ngf: int) -> object:
        super().__init__()
        
        #self.color_net = submodel.UNet(in_channel, 3, ngf)
        self.color_net = UNet(in_channel, out_channel=3, ngf=ngf, upconv=False, norm=True)
        self.depth_net = SUNet(in_channel, out_channel=1, ngf=ngf, upconv=False, norm=True)
        
    
    def forward(self, color_img: torch.tensor, depth_img: torch.tensor) -> torch.tensor:
        #color_front = self.color_net(torch.cat((color_img, depth_img), dim=1),
        #                         inter_mode='bilinear')
        color_front = self.color_net(torch.cat((color_img, depth_img), dim=1))
        depth_front = self.depth_net(torch.cat((color_img, depth_img), dim=1))
        #print(depth_front.shape)

        return color_front, depth_front