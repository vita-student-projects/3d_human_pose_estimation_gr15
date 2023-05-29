from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import logging
import torchvision.models.resnet as resnet
import torchvision.transforms.functional as F
import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from utils import BHWC_to_BCHW, copy_state_dict
from models.CoordConv import get_coord_maps
import config
from config import args
from models.basic_modules import BasicBlock,Bottleneck,HighResolutionModule

import torchvision.models as models


BN_MOMENTUM = 0.1

class EfficientNetRomp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        net = models.efficientnet_v2_s(weights=None).cuda()
        self.feature_extractor = net.features
        self.deconv_layers = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(1280, 256, (3,3,), 2, 1, 1),
            torch.nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(256, 128, (3,3,), 2, 1, 1),
            torch.nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(128, 64, (3,3,), 2, 1, 1),
            torch.nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            torch.nn.ReLU(inplace=True)
        )
        self.backbone_channels = 64
    
    def forward(self, x: torch.Tensor):
        # print(self.deconv_layers[0].weight[:1,:1,:1,:1])
        x = x.float().permute(0,3,1,2)
        # print("B", x.requires_grad)
        x = self.feature_extractor(x)
        # print("C", x.requires_grad)
        # print("T?", self.training)


        x = self.deconv_layers(x)
        return x

    def image_preprocess(self, x):
        x = BHWC_to_BCHW(x)/255.
        #x = F.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],inplace=True).contiguous() # for pytorch version>1.8.0
        x = torch.stack(list(map(lambda x:F.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],inplace=False),x)))
        #x = ((BHWC_to_BCHW(x)/ 255.) * 2.0 - 1.0).contiguous()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
    
    def load_pretrain_params(self):
        if os.path.exists(args().resnet_pretrain):
            try:
                
                net = models.efficientnet_v2_s(weights=None).cuda()
                print(self.deconv_layers[0].weight[:1,:1,:1,:1])

                self.feature_extractor = net.features
                success_layer = copy_state_dict(self.state_dict(), torch.load(args().effnet_pretrain), prefix = 'backbone.', fix_loaded=False)
                print(self.deconv_layers[0].weight[:1,:1,:1,:1])

                logging.info("LOADED EffNet")
                # raise Exception("Doogo")                
            except AttributeError:
                logging.error("No pretrained network for EffNet backbone in config - Starting from scratch ...")
                net = models.efficientnet_v2_s(models.efficientnet.EfficientNet_V2_S_Weights).cuda()
                self.feature_extractor = net.features
                # self.init_weights()
            except FileNotFoundError:
                logging.warn("Pretrained parameters for EffNet backbone not found on disk ('{}') - Starting from scratch ...".format(args().effnet_pretrain))
                net = models.efficientnet_v2_s(weights=models.efficientnet.EfficientNet_V2_S_Weights).cuda()
                self.feature_extractor = net.features
                # self.init_weights()


    

if __name__ == '__main__':
    args().pretrain = 'spin'
    model = EfficientNetRomp().cuda()
    a=model(torch.rand(2,512,512,3).cuda())
    for i in a:
        print(i.shape)