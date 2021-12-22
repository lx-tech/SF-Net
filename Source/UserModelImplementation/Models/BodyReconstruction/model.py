import torch
import torch.nn as nn
import sys
sys.path.append("./Source/UserModelImplementation/Models/BodyReconstruction")
import submodel 

class Model(nn.Module):
    """docstring for Model"""

    def __init__(self, in_channel, out_channel, ngf, upconv=False, norm=False):
        super().__init__()
        
        self.color_net = submodel.UNet(in_channel, out_channel, ngf)
        #self.depth_net = submodel.UNet(in_channel, 1 ,ngf)
    
    def forward(self, color_img: torch.tensor, depth_img: torch.tensor) -> torch.tensor:
        color_front = self.color_net(color_img, inter_mode='bilinear')
        #depth_front = self.depth_net(depth_img, inter_mode='bilinear')

        return color_front