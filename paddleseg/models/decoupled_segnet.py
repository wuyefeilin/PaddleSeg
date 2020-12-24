# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.models.backbones import resnet_vd
from paddleseg.models import deeplab
from paddleseg.utils import utils


@manager.MODELS.add_component
class DecoupledSegNet(nn.Layer):
    """
    The DecoupledSegNet implementation based on PaddlePaddle.

    The original article refers to
    , et, al. ""
    ()

    Args:
        num_classes (int): The unique number of target classes.
        backbone (paddle.nn.Layer): Backbone network, currently support Resnet50_vd/Resnet101_vd.
        backbone_indices (tuple, optional): Two values in the tuple indicate the indices of output of backbone.
           Default: (0, 3).
        aspp_ratios (tuple, optional): The dilation rate using in ASSP module.
            If output_stride=16, aspp_ratios should be set as (1, 6, 12, 18).
            If output_stride=8, aspp_ratios is (1, 12, 24, 36).
            Default: (1, 6, 12, 18).
        aspp_out_channels (int, optional): The output channels of ASPP module. Default: 256.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(0, 3),
                 aspp_ratios=(1, 6, 12, 18),
                 aspp_out_channels=256,
                 align_corners=False,
                 pretrained=None):
        super().__init__()
        self.backbone = backbone
        backbone_channels = self.backbone.feat_channels
        self.head = DecoupledSegNetHead(num_classes, backbone_indices,
                                        backbone_channels, aspp_ratios,
                                        aspp_out_channels, align_corners)
        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        feat_list = self.backbone(x)
        logit_list = self.head(feat_list)

        seg_logit, body_logit, edge_logit = [
            F.interpolate(
                logit,
                x.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]

        # deeplabv3
        # seg_logit = [
        #     F.interpolate(
        #         logit,
        #         x.shape[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners) for logit in logit_list
        # ]
        # return [seg_logit]
        return [seg_logit, body_logit, edge_logit, (seg_logit, edge_logit)]

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class DecoupledSegNetHead(nn.Layer):
    """
    """

    def __init__(self, num_classes, backbone_indices, backbone_channels,
                 aspp_ratios, aspp_out_channels, align_corners):
        super().__init__()
        self.backbone_indices = backbone_indices
        self.align_corners = align_corners
        self.aspp = layers.ASPPModule(
            aspp_ratios=aspp_ratios,
            in_channels=backbone_channels[backbone_indices[1]],
            out_channels=aspp_out_channels,
            align_corners=align_corners,
            image_pooling=True)

        # self.bot_aspp = nn.Conv2D(aspp_out_channels * 5, 256, kernel_size=1, bias_attr=False)
        self.bot_fine = nn.Conv2D(
            backbone_channels[backbone_indices[0]], 48, 1, bias_attr=False)
        # decoupled
        self.squeeze_body_edge = SqueezeBodyEdge(
            256, align_corners=self.align_corners)
        self.edge_fusion = nn.Conv2D(256 + 48, 256, 1, bias_attr=False)
        self.sigmoid_edge = nn.Sigmoid()
        self.edge_out = nn.Sequential(
            layers.ConvBNReLU(
                in_channels=256,
                out_channels=48,
                kernel_size=3,
                bias_attr=False), nn.Conv2D(48, 1, 1, bias_attr=False))
        self.dsn_seg_body = nn.Sequential(
            layers.ConvBNReLU(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                bias_attr=False), nn.Conv2D(
                    256, num_classes, 1, bias_attr=False))

        self.final_seg = nn.Sequential(
            layers.ConvBNReLU(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                bias_attr=False),
            layers.ConvBNReLU(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                bias_attr=False),
            nn.Conv2D(256, num_classes, kernel_size=1, bias_attr=False))

        # # deeplabv3 输出
        # self.cls = nn.Conv2D(
        #     in_channels=aspp_out_channels,
        #     out_channels=num_classes,
        #     kernel_size=1)

    def forward(self, feat_list):
        fine_fea = feat_list[self.backbone_indices[0]]
        fine_size = fine_fea.shape
        x = feat_list[self.backbone_indices[1]]
        # print('backbone', x.shape)
        # x = self.aspp(x) # 256c
        # print('aspp out', x.shape)
        # aspp = self.bot_aspp(x)
        aspp = self.aspp(x)

        # decoupled
        # print('aspp', aspp.shape)
        seg_body, seg_edge = self.squeeze_body_edge(aspp)
        # Edge presevation and edge out
        fine_fea = self.bot_fine(fine_fea)
        seg_edge = F.interpolate(
            seg_edge,
            fine_size[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_edge = self.edge_fusion(paddle.concat([seg_edge, fine_fea], axis=1))
        seg_edge_out = self.edge_out(seg_edge)
        seg_edge_out = self.sigmoid_edge(seg_edge_out)  # seg_edge output
        # Body out
        # print('seg_body', seg_body.shape)
        seg_body_out = self.dsn_seg_body(seg_body)

        # seg_final out
        seg_out = seg_edge + F.interpolate(
            seg_body,
            fine_size[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        aspp = F.interpolate(
            aspp,
            fine_size[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_out = paddle.concat([aspp, seg_out], axis=1)
        seg_final_out = self.final_seg(seg_out)

        return [seg_final_out, seg_body_out, seg_edge_out]

        # return [self.cls(aspp)]


class SqueezeBodyEdge(nn.Layer):
    def __init__(self, inplane, align_corners=False):
        super().__init__()
        self.align_corners = align_corners
        self.down = nn.Sequential(
            layers.ConvBNReLU(
                inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            layers.ConvBNReLU(
                inplane, inplane, kernel_size=3, groups=inplane, stride=2))
        self.flow_make = nn.Conv2D(
            inplane * 2, 2, kernel_size=3, padding='same', bias_attr=False)

    def forward(self, x):
        size = x.shape[2:]
        seg_down = self.down(x)
        seg_down = F.interpolate(
            seg_down,
            size=size,
            mode='bilinear',
            align_corners=self.align_corners)
        flow = self.flow_make(paddle.concat([x, seg_down], axis=1))
        seg_flow_warp = self.flow_warp(x, flow, size)
        seg_edge = x - seg_flow_warp
        return seg_flow_warp, seg_edge

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.shape
        norm = paddle.to_tensor([[[[out_w, out_h]]]], dtype='float32')
        h_grid = paddle.linspace(-1.0, 1.0, out_h).reshape([-1, 1])
        h_grid = paddle.concat([h_grid] * out_w, axis=1)
        w_grid = paddle.linspace(-1.0, 1.0, out_w).reshape([1, -1])
        w_grid = paddle.concat([w_grid] * out_h, axis=0)
        grid = paddle.concat([w_grid.unsqueeze(2), h_grid.unsqueeze(2)], axis=2)

        grid = paddle.concat([grid.unsqueeze(0)] * n, axis=0)
        grid = grid + paddle.transpose(flow, (0, 2, 3, 1)) / norm

        output = F.grid_sample(input, grid)
        return output


if __name__ == '__main__':
    # x = paddle.rand((1, 3, 512, 512))
    # backbone = resnet_vd.ResNet50_vd(output_stride=8)
    # model = DecoupledSegNet(
    #     num_classes=2,
    #     backbone=backbone,
    #     backbone_indices=(0, 3),
    #     aspp_ratios=(1, 12, 24, 36),
    #     aspp_out_channels=256,
    #     align_corners=False,
    #     pretrained=None)
    # out = model(x)

    input = paddle.rand((1, 1, 4, 4))
    flow = paddle.rand((1, 2, 4, 4))
    size = [4, 4]

    out_h, out_w = size
    n, c, h, w = input.shape
    norm = paddle.to_tensor([[[[out_w, out_h]]]], dtype='float32')
    h_grid = paddle.linspace(-1.0, 1.0, out_h).reshape([-1, 1])
    h_grid = paddle.concat([h_grid] * out_w, axis=1)
    w_grid = paddle.linspace(-1.0, 1.0, out_w).reshape([1, -1])
    w_grid = paddle.concat([w_grid] * out_h, axis=0)
    grid = paddle.concat([w_grid.unsqueeze(2), h_grid.unsqueeze(2)], axis=2)

    grid = paddle.concat([grid.unsqueeze(0)] * n, axis=0)
    grid = grid + paddle.transpose(flow, (0, 2, 3, 1)) / norm

    output = F.grid_sample(input, grid)
    print('input')
    print(input)
    print('flow')
    print(flow)
    print('grid')
    print(grid)
    print('output')
    print(output)