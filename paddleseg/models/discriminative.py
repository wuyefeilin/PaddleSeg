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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg import utils
from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers


@manager.MODELS.add_component
class InstanceFCN(nn.Layer):
    """
    A simple implementation for Instance Segmentation based on PaddlePaddle.

    The original article refers to
    Bert De Brabandere, et, al. "Semantic Instance Segmentation with a Discriminative Loss Function"
    (https://arxiv.org/pdf/1708.02551.pdf).

    Args:
        num_classes (int): The unique number of target classes.
        backbone (paddle.nn.Layer): Backbone networks.
        backbone_indices (tuple, optional): The values in the tuple indicate the indices of output of backbone.
            Default: (-1, ).
        channels (int, optional): The channels between conv layer and the last layer of FCNHead.
            If None, it will be the number of channels of input features. Default: None.
        spatial_dim (int, optional): The spatial dimentions for instance representation. Default: 8.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(-1, ),
                 channels=None,
                 spatial_dim=8,
                 align_corners=False,
                 pretrained=None):
        super(InstanceFCN, self).__init__()

        self.backbone = backbone
        backbone_channels = [
            backbone.feat_channels[i] for i in backbone_indices
        ]

        self.head = FCNHead(num_classes, backbone_indices, backbone_channels,
                            channels, spatial_dim)

        self.align_corners = align_corners
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.init_weight()

    def forward(self, x):
        feat_list = self.backbone(x)
        seg_pred, instance_pred = self.head(feat_list)
        seg_pred = F.interpolate(
            seg_pred,
            x.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        instance_pred = F.interpolate(
            instance_pred,
            x.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.training:
            return seg_pred, instance_pred
        else:
            instance_map = self.clustering_v2(seg_pred, instance_pred)
            seg_pred = F.softmax(seg_pred)
            # seg_pred: nchw, instance_map: nhw
            return seg_pred, instance_map

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def clustering(self, seg_pred, instance_pred):
        n, c, h, w = seg_pred.shape
        _, s, _, _ = instance_pred.shape
        instance_pred = paddle.transpose(instance_pred, (0, 2, 3, 1))
        instance_map = paddle.zeros((n, h, w), dtype='int64')
        for i in range(n):
            for _ in range(1, c):
                instance_id = 1
                instance_maps = []
                while True:
                    segmask = (seg_pred[i] == _).astype('int32')
                    insmask = (instance_map[i] == 0).astype('int32')
                    indexs = paddle.nonzero(segmask * insmask)
                    if indexs.shape[0] == 0:
                        break
                    center = instance_pred[
                        i, int(indexs[0][0]
                               ), int(indexs[0][1])]
                    last_center = 0

                    while paddle.norm(
                            center - last_center, p=2).numpy()[0] > 1e-5:
                        distance = instance_pred[i] - center
                        distance = paddle.norm(distance, p=2, axis=2)
                        dismask = (distance < 2.5).astype('int32')
                        last_center = center
                        cnt = paddle.sum(dismask * segmask * insmask)
                        center = paddle.sum(
                            instance_pred[i] *
                            (dismask * segmask * insmask).unsqueeze(2),
                            axis=[0, 1]) / cnt

                    dis_indexs = paddle.nonzero(dismask * segmask * insmask)
                    instance_map[i] = paddle.scatter_nd_add(
                        instance_map[i], dis_indexs,
                        paddle.to_tensor(
                            [instance_id + _ * 1000] * len(dis_indexs)))
                    instance_id += 1

        return instance_map

    def clustering_v2(self, seg_pred, instance_pred):
        n, c, h, w = seg_pred.shape
        _, s, _, _ = instance_pred.shape
        seg_pred = paddle.argmax(seg_pred, axis=1)
        instance_pred = paddle.transpose(instance_pred, (0, 2, 3, 1))
        instance_map = paddle.zeros((n, h, w), dtype='int64')
        for i in range(n):
            for _ in range(1, c):
                instance_id = 1
                segmask = (seg_pred[i] == _).astype('int32')
                segcnt = paddle.sum(segmask).numpy()[0]
                instance_maps = []
                counter = 0

                while segcnt - paddle.sum(
                    (instance_map[i] !=
                     0).astype('int32')).numpy()[0] > 200 and counter < 20:

                    insmask = (instance_map[i] == 0).astype('float32')
                    indexs = paddle.nonzero(segmask * insmask)
                    if indexs.shape[0] == 0:
                        break

                    _idx = int(paddle.randint(indexs.shape[0]).numpy()[0])

                    center = instance_pred[
                        i, int(indexs[_idx][0]
                               ), int(indexs[_idx][1])]
                    last_center = 0

                    while paddle.norm(
                            center - last_center, p=2).numpy()[0] > 1e-4:
                        distance = instance_pred[i] - center
                        distance = paddle.norm(distance, p=2, axis=2)**2
                        dismask = (distance < 1.5).astype('int32')
                        last_center = center
                        cnt = paddle.sum(dismask * segmask * insmask)
                        center = paddle.sum(
                            instance_pred[i] *
                            (dismask * segmask * insmask).unsqueeze(2),
                            axis=[0, 1]) / cnt

                    distance = instance_pred[i] - center
                    distance = paddle.norm(distance, p=2, axis=2)**2
                    dismask = (distance < 1.5).astype('int32')
                    iop = (paddle.sum((1 - insmask) * dismask * segmask) /
                           paddle.sum(dismask * segmask)).numpy()[0]

                    if iop < 0.5:
                        dis_indexs = paddle.nonzero(dismask * segmask * insmask)
                        instance_map[i] = paddle.scatter_nd_add(
                            instance_map[i], dis_indexs,
                            paddle.to_tensor(
                                [instance_id + _ * 1000] * len(dis_indexs)))
                        instance_id += 1
                        counter = 0
                    else:
                        counter += 1

        return instance_map


class FCNHead(nn.Layer):
    """
    A simple implementation for FCNHead based on PaddlePaddle

    Args:
        num_classes (int): The unique number of target classes.
        backbone_indices (tuple, optional): The values in the tuple indicate the indices of output of backbone.
            Default: (-1, ).
        backbone_channels (tuple): The same length with "backbone_indices". It indicates the channels of corresponding index.
        channels (int, optional): The channels between conv layer and the last layer of FCNHead.
            If None, it will be the number of channels of input features. Default: None.
        spatial_dim (int, optional): The spatial dimentions for instance representation. Default: 8.
    """

    def __init__(self,
                 num_classes,
                 backbone_indices=(-1, ),
                 backbone_channels=(270, ),
                 channels=None,
                 spatial_dim=8):
        super(FCNHead, self).__init__()

        self.num_classes = num_classes
        self.backbone_indices = backbone_indices
        self.spatial_dim = spatial_dim
        if channels is None:
            channels = backbone_channels[0]

        self.conv_1 = layers.ConvBNReLU(
            in_channels=backbone_channels[0],
            out_channels=channels,
            kernel_size=1,
            padding='same',
            stride=1)
        self.seg_head = nn.Conv2D(
            in_channels=channels,
            out_channels=self.num_classes,
            kernel_size=1,
            stride=1,
            padding=0)
        self.instance_head = nn.Conv2D(
            in_channels=channels,
            out_channels=spatial_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.init_weight()

    def forward(self, feat_list):
        logit_list = []
        x = feat_list[self.backbone_indices[0]]
        x = self.conv_1(x)

        seg_pred = self.seg_head(x)
        instance_pred = self.instance_head(x)

        return seg_pred, instance_pred

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.normal_init(layer.weight, std=0.001)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)
