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

import numpy as np
import paddle
import paddle.nn.functional as F

from paddleseg.utils import ins_metrics, Timer, calculate_eta, logger, progbar

np.set_printoptions(suppress=True)


# TODO multi gpus evaluation
def evaluate(model, eval_dataset, num_workers=0):
    model.eval()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank
    if nranks > 1:
        # Initialize parallel environment if not done.
        if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
        ):
            paddle.distributed.init_parallel_env()
    # batch_sampler = paddle.io.DistributedBatchSampler(
    #     eval_dataset, batch_size=1, shuffle=False, drop_last=False)
    batch_sampler = paddle.io.BatchSampler(
        eval_dataset, batch_size=1, shuffle=False, drop_last=False)
    loader = paddle.io.DataLoader(
        eval_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        return_list=True,
    )

    total_iters = len(loader)
    metric_50 = ins_metrics.AveragePrecision(
        eval_dataset.num_classes - 1,
        overlaps=0.5)  # exclude 0 since it is background
    metric = ins_metrics.AveragePrecision(
        eval_dataset.num_classes - 1, overlaps=list(np.arange(
            0.5, 1.0, 0.05)))  # exclude 0 since it is background

    logger.info("Start evaluating (total_samples={}, total_iters={})...".format(
        len(eval_dataset), total_iters))
    progbar_val = progbar.Progbar(target=total_iters, verbose=1)
    timer = Timer()
    for iter, data in enumerate(loader):
        reader_cost = timer.elapsed_time()
        im = data[0]
        seg_label, ins_label = data[1].astype('int64'), data[2].astype('int64')
        seg_pred, ins_pred = model(im)

        seg_label = seg_label.numpy()
        ins_label = ins_label.numpy()
        seg_pred = seg_pred.numpy()
        ins_pred = ins_pred.numpy()

        ignore_mask = seg_label == eval_dataset.ignore_index
        for i in range(im.shape[0]):
            gts = ins_metrics.convert_gt_map(seg_label[i], ins_label[i])
            preds = ins_metrics.convert_pred_map(seg_pred[i], ins_pred[i])
            metric_50.compute(preds, gts, ignore_mask=ignore_mask[i])
            metric.compute(preds, gts, ignore_mask=ignore_mask[i])

        batch_cost = timer.elapsed_time()
        timer.restart()
        if local_rank == 0:
            progbar_val.update(iter + 1, [('batch_cost', batch_cost),
                                          ('reader cost', reader_cost)])

    ap = metric.cal_ap()
    map = metric.cal_map()
    ap_50 = metric_50.cal_ap()
    map_50 = metric_50.cal_map()
    logger.info("[EVAL] #Images={} mAP={:.4f} mAP_50={:.4f}".format(
        len(eval_dataset), map, map_50))
    logger.info("[EVAL] Class AP: \n" + str(np.round(ap, 4)))
    logger.info("[EVAL] Class AP_50: \n" + str(np.round(ap_50, 4)))
    return map, map_50, ap, ap_50
