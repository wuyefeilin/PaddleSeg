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

import os

import cv2
import numpy as np
import paddle

from paddleseg import utils
from paddleseg.utils import logger, progbar


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def predict(model,
            model_path,
            transforms,
            image_list,
            image_dir=None,
            save_dir='output'):
    """
    predict and visualize the image_list.

    Args:
        model (nn.Layer): Used to predict for input image.
        model_path (str): The path of pretrained model.
        transforms (transform.Compose): Preprocess for input image.
        image_list (list): A list of images to be predicted.
        image_dir (str): The directory of the images to be predicted. Default: None.
        save_dir (str): The directory to save the visualized results. Default: 'output'.

    """
    para_state_dict = paddle.load(model_path)
    model.set_dict(para_state_dict)
    model.eval()

    added_saved_dir = os.path.join(save_dir, 'added_prediction')
    pred_saved_dir = os.path.join(save_dir, 'pseudo_color_prediction')

    logger.info("Start to predict...")
    progbar_pred = progbar.Progbar(target=len(image_list), verbose=1)
    for i, im_path in enumerate(image_list):
        im = cv2.imread(im_path)
        im, _ = transforms(im)
        im = im[np.newaxis, ...]
        im = paddle.to_tensor(im)

        seg_pred, ins_pred = model(im)
        # convert value of ins_pre 1001, 1002, ...,2001, ... to 1,2,3,...
        print(ins_pred.shape)
        ins_pred = paddle.squeeze(ins_pred, axis=0)
        ins_pred = ins_pred.numpy()
        ins_cnt = np.unique(ins_pred)
        for _idx, id in enumerate(ins_cnt):
            if id == 0:
                continue
            ins_pred[ins_pred == id] = _idx

        ins_pred = ins_pred.astype('uint8')

        # get the saved name
        if image_dir is not None:
            im_file = im_path.replace(image_dir, '')
        else:
            im_file = os.path.basename(im_path)
        if im_file[0] == '/':
            im_file = im_file[1:]

        # save added image
        added_image = utils.visualize.visualize(im_path, ins_pred, weight=0.6)
        added_image_path = os.path.join(added_saved_dir, im_file)
        mkdir(added_image_path)
        cv2.imwrite(added_image_path, added_image)

        # save pseudo color prediction
        pred_mask = utils.visualize.get_pseudo_color_map(ins_pred)
        pred_saved_path = os.path.join(pred_saved_dir,
                                       im_file.rsplit(".")[0] + ".png")
        mkdir(pred_saved_path)
        pred_mask.save(pred_saved_path)

        progbar_pred.update(i + 1)
