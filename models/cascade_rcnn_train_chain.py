from __future__ import division

import numpy as np

import chainer

from models.rpn import rpn_loss
from models.bbox_head import bbox_head_loss_pre, bbox_head_loss_post


class CascadeRCNNTrainChain(chainer.Chain):

    def __init__(self, model):
        super(CascadeRCNNTrainChain, self).__init__()
        with self.init_scope():
            self.model = model

    def forward(self, imgs, bboxes, labels):
        B = len(imgs)
        pad_size = np.array(
            [im.shape[1:] for im in imgs]).max(axis=0)
        pad_size = (
            np.ceil(
                pad_size / self.model.stride) * self.model.stride).astype(int)
        x = np.zeros(
            (len(imgs), 3, pad_size[0], pad_size[1]), dtype=np.float32)
        for i, img in enumerate(imgs):
            _, H, W = img.shape
            x[i, :, :H, :W] = img
        x = self.xp.array(x)

        bboxes = [self.xp.array(bbox) for bbox in bboxes]
        labels = [self.xp.array(label) for label in labels]
        sizes = [img.shape[1:] for img in imgs]

        with chainer.using_config('train', False):
            hs = self.model.extractor(x)

        B = len(imgs)
        rpn_locs, rpn_confs = self.model.rpn(hs)
        anchors = self.model.rpn.anchors(h.shape[2:] for h in hs)
        rpn_loc_loss, rpn_conf_loss = rpn_loss(
            rpn_locs, rpn_confs, anchors, sizes, bboxes)

        rois, roi_indices = self.model.rpn.decode(
            rpn_locs, rpn_confs, anchors, x.shape)
        rois = self.xp.vstack([rois] + bboxes)
        roi_indices = self.xp.hstack(
            [roi_indices]
            + [self.xp.array((i,) * len(bbox))
               for i, bbox in enumerate(bboxes)])

        rois, roi_indices = self.model.bbox_heads[0].distribute(
            rois, roi_indices)

        loss = 0
        report_losses = {
            'loss': 0, 'loss/bbox_head/loc': 0, 'loss/bbox_head/conf': 0,
            'loss/rpn/loc': rpn_loc_loss, 'loss/rpn/conf': rpn_conf_loss
        }
        for i, bbox_head in enumerate(self.model.bbox_heads):
            rois, roi_indices, head_gt_locs, head_gt_labels = \
                bbox_head_loss_pre(
                    rois, roi_indices, bbox_head.std, bboxes, labels,
                    bbox_head.thresh)
            head_locs, head_confs = bbox_head(hs, rois, roi_indices)
            head_loc_loss, head_conf_loss = bbox_head_loss_post(
                head_locs, head_confs,
                roi_indices, head_gt_locs, head_gt_labels, B)
            bbox = bbox_head.decode_bbox(
                rois, roi_indices, head_locs, sizes)
            last_idx = 0
            for j, ri in enumerate(roi_indices):
                rois[j] = bbox[last_idx:last_idx + ri.shape[0]]
                last_idx += ri.shape[0]

            loss += (rpn_loc_loss + rpn_conf_loss +
                     head_loc_loss + head_conf_loss)
            self._update_report(
                report_losses, head_loc_loss, head_conf_loss, i)
        chainer.reporter.report(report_losses, self)
        return loss

    @staticmethod
    def _update_report(report, loss_loc, loss_conf, i):
        report['loss/bbox_head/loc'] += loss_loc
        report['loss/bbox_head/conf'] += loss_conf
        report['loss'] += (loss_loc + loss_conf)
        report['loss/bbox_head/stage{}/loc'.format(i)] = loss_loc
        report['loss/bbox_head/stage{}/conf'.format(i)] = loss_conf
