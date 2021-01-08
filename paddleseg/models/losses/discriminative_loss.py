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
from paddle import nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class DiscriminativeLoss(nn.Layer):
    def __init__(self,
                 num_classes=9,
                 norm=2,
                 ignore_index=255,
                 delta_var=0.5,
                 delta_dist=1.5,
                 alpha=1,
                 beta=1,
                 gamma=0.001):
        super(DiscriminativeLoss, self).__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.norm = norm
        self.num_classes = num_classes
        self.ESP = 1e-5

    def forward(self, logit, label, seglabel):
        n = logit.shape[0]
        loss = 0
        for i in range(n):
            for cls in range(self.num_classes):
                mask = (seglabel[i] == cls).astype('int32')
                mask.stop_gradient = True
                logit_n_c = logit[i] * mask
                label_n_c = label[i] * mask
                instance_means = self._cluster_mean(logit_n_c, label_n_c)
                if instance_means is None:
                    continue
                var_term = self._variance_term(logit_n_c, label_n_c,
                                               instance_means)
                dis_term = self._distance_term(instance_means)
                reg_term = self._regularization_term(instance_means)
                loss += self.alpha * var_term + self.beta * dis_term + self.gamma * reg_term
        return loss / n

    def _variance_term(self, logit, label, instance_means):
        instance_cnt = instance_means.shape[1]
        var_term = 0
        instance_ids = paddle.unique(label).numpy()
        for _idx in range(instance_cnt):
            instance_mask = (label == int(
                instance_ids[_idx + 1])).astype('int32')
            instance_mean = instance_means[:, _idx].unsqueeze(1).unsqueeze(2)
            variance = paddle.norm(logit - instance_mean, p=self.norm, axis=0)
            variance = paddle.clip(variance - self.delta_var, min=0)**2
            cnt = paddle.sum(instance_mask)
            var_term += paddle.sum(variance * instance_mask) / cnt
        return var_term / instance_cnt

    def _distance_term(self, instance_means):
        c, i = instance_means.shape
        instance_means = instance_means.unsqueeze(2).expand([c, i, i])
        other_means = instance_means.transpose([0, 2, 1])
        diff = instance_means - other_means
        distance = paddle.norm(diff, p=self.norm, axis=0)
        margin = self.delta_dist * 2 * (1 - paddle.eye(i))
        dis_term = paddle.sum(paddle.clip(margin - distance, min=0)**2)
        return dis_term / (2 * i * (i - 1) + self.ESP)

    def _regularization_term(self, instance_means):
        reg_term = 0
        for mean in instance_means:
            reg_term += paddle.norm(mean, p=self.norm)
        return reg_term / instance_means.shape[1]

    def _cluster_mean(self, logit, label):
        instance_cnt = paddle.unique(label).shape[0]

        instance_means = []
        instance_ids = paddle.unique(label).numpy()
        for _idx in range(instance_cnt):
            if int(instance_ids[_idx]) == 0:
                continue

            instance_mask = (label == int(instance_ids[_idx])).astype('int32')
            instance_pixels = paddle.sum(instance_mask)
            instance_mean = paddle.sum(
                logit * instance_mask, axis=[1, 2]) / instance_pixels
            instance_mean = instance_mean.unsqueeze(1)
            instance_means.append(instance_mean)

        if len(instance_means) == 0:
            return None

        instance_means = paddle.concat(instance_means, axis=1)
        return instance_means
