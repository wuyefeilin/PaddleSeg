# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob

import numpy as np
from PIL import Image

from paddleseg.datasets import Dataset
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose


@manager.DATASETS.add_component
class Cityscapes(Dataset):
    """
    Cityscapes dataset `https://www.cityscapes-dataset.com/`.
    The folder structure is as follow:

        cityscapes
        |
        |--leftImg8bit
        |  |--train
        |  |--val
        |  |--test
        |
        |--gtFine
        |  |--train
        |  |--val
        |  |--test

    Make sure there are **labelTrainIds.png in gtFine directory. If not, please run the conver_cityscapes.py in tools.

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): Cityscapes dataset directory.
        mode (str): Which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
    """

    ID_MAP = {
        0: 255,
        1: 255,
        2: 255,
        3: 255,
        4: 255,
        5: 255,
        6: 255,
        7: 0,
        8: 0,
        9: 255,
        10: 255,
        11: 0,
        12: 0,
        13: 0,
        14: 255,
        15: 255,
        16: 255,
        17: 0,
        18: 255,
        19: 0,
        20: 0,
        21: 0,
        22: 0,
        23: 0,
        24: 1,
        25: 2,
        26: 3,
        27: 4,
        28: 5,
        29: 255,
        30: 255,
        31: 6,
        32: 7,
        33: 8
    }

    def __init__(self,
                 transforms,
                 dataset_root,
                 mode='train',
                 load_instance='True'):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.file_list = list()
        mode = mode.lower()
        self.mode = mode
        self.num_classes = 9
        self.ignore_index = 255
        self.load_instance = load_instance

        if mode not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        img_dir = os.path.join(self.dataset_root, 'leftImg8bit')
        label_dir = os.path.join(self.dataset_root, 'gtFine')
        if self.dataset_root is None or not os.path.isdir(
                self.dataset_root) or not os.path.isdir(
                    img_dir) or not os.path.isdir(label_dir):
            raise ValueError(
                "The dataset is not Found or the folder structure is nonconfoumance."
            )

        pattern = '*_gtFine_instanceIds.png' if load_instance else '*_gtFine_labelTrainIds.png'
        label_files = sorted(
            glob.glob(os.path.join(label_dir, mode, '*', pattern)))
        img_files = sorted(
            glob.glob(os.path.join(img_dir, mode, '*', '*_leftImg8bit.png')))

        self.file_list = [[
            img_path, label_path
        ] for img_path, label_path in zip(img_files, label_files)]

    def __getitem__(self, idx):
        image_path, label_path = self.file_list[idx]
        if self.mode == 'test':
            im, _ = self.transforms(im=image_path)
            im = im[np.newaxis, ...]
            return im, image_path
        elif self.mode == 'val':
            im, _ = self.transforms(im=image_path)
            label = np.asarray(Image.open(label_path))
            label = label[np.newaxis, :, :]
            seg_map, instance_map = self.gen_instance_label(label)
            return im, seg_map, instance_map
        else:
            im, label = self.transforms(im=image_path, label=label_path)
            seg_map, instance_map = self.gen_instance_label(label)
            return im, seg_map, instance_map

    def gen_instance_label(self, label):
        instance_mask = (label > 1000).astype('int32')
        seg_mask = 1 - instance_mask
        seg_map = seg_mask * label + (instance_mask * label) // 1000
        instance_cnt = np.unique(label * instance_mask)
        instance_map = label * instance_mask
        for _idx, id in enumerate(instance_cnt):
            instance_map[instance_map == id] = _idx

        for key, value in Cityscapes.ID_MAP.items():
            seg_map[seg_map == key] = value
        return seg_map, instance_map
