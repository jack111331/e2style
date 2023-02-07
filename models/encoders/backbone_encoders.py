import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module

from models.encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE
from models.stylegan2.model import EqualConv2d, ScaledLeakyReLU

class BackboneEncoderFirstStage(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderFirstStage, self).__init__()
        # print('Using BackboneEncoderFirstStage')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        self.output_layer_3 = Sequential(BatchNorm2d(256),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(256 * 7 * 7, 512 * 9))
        self.output_layer_4 = Sequential(BatchNorm2d(128),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(128 * 7 * 7, 512 * 5))
        self.output_layer_5 = Sequential(BatchNorm2d(64),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(64 * 7 * 7, 512 * 4))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        self.modulelist = list(self.body)

    def forward(self, x):
        x = self.input_layer(x)
        for l in self.modulelist[:3]:
          x = l(x)
        lc_part_4 = self.output_layer_5(x).view(-1, 4, 512)
        for l in self.modulelist[3:7]:
          x = l(x)
        lc_part_3 = self.output_layer_4(x).view(-1, 5, 512)
        for l in self.modulelist[7:21]:
          x = l(x)
        lc_part_2 = self.output_layer_3(x).view(-1, 9, 512)

        x = torch.cat((lc_part_2, lc_part_3, lc_part_4), dim=1)
        return x

class BackboneEncoderRefineStage(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderRefineStage, self).__init__()
        # print('Using BackboneEncoderRefineStage')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(6, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        self.output_layer_3 = Sequential(BatchNorm2d(256),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(256 * 7 * 7, 512 * 9))
        self.output_layer_4 = Sequential(BatchNorm2d(128),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(128 * 7 * 7, 512 * 5))
        self.output_layer_5 = Sequential(BatchNorm2d(64),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(64 * 7 * 7, 512 * 4))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        self.modulelist = list(self.body)

    def forward(self, x, first_stage_output_image):
        x = torch.cat((x,first_stage_output_image), dim=1)
        x = self.input_layer(x)
        for l in self.modulelist[:3]:
          x = l(x)
        lc_part_4 = self.output_layer_5(x).view(-1, 4, 512)
        for l in self.modulelist[3:7]:
          x = l(x)
        lc_part_3 = self.output_layer_4(x).view(-1, 5, 512)
        for l in self.modulelist[7:21]:
          x = l(x)
        lc_part_2 = self.output_layer_3(x).view(-1, 9, 512)

        x = torch.cat((lc_part_2, lc_part_3, lc_part_4), dim=1)
        return x

# Consultation encoder
class ResidualEncoder(Module):
    def __init__(self,  opts=None):
        super(ResidualEncoder, self).__init__()
        self.conv_layer1 = Sequential(Conv2d(3, 32, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(32),
                                      PReLU(32))

        self.conv_layer2 =  Sequential(*[bottleneck_IR(32,48,2), bottleneck_IR(48,48,1), bottleneck_IR(48,48,1)])

        self.conv_layer3 =  Sequential(*[bottleneck_IR(48,64,2), bottleneck_IR(64,64,1), bottleneck_IR(64,64,1)])

        self.condition_scale3 = nn.Sequential(
                    EqualConv2d(64, 512, 3, stride=1, padding=1, bias=True ),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(512, 512, 3, stride=1, padding=1, bias=True ))

        self.condition_shift3 = nn.Sequential(
                    EqualConv2d(64, 512, 3, stride=1, padding=1, bias=True ),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(512, 512, 3, stride=1, padding=1, bias=True ))  
                    
    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def forward(self, x):
        conditions = []
        feat1 = self.conv_layer1(x)
        feat2 = self.conv_layer2(feat1)
        feat3 = self.conv_layer3(feat2)

        scale = self.condition_scale3(feat3)
        scale = torch.nn.functional.interpolate(scale, size=(64, 32) , mode='bilinear')
        conditions.append(scale.clone())
        shift = self.condition_shift3(feat3)
        shift = torch.nn.functional.interpolate(shift, size=(64, 32) , mode='bilinear')
        conditions.append(shift.clone())  
        return conditions