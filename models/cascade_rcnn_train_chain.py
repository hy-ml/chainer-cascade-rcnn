from __future__ import division

import numpy as np

import chainer
import chainer.functions as F

from chainercv.links.model.fpn import bbox_head_loss_post
from chainercv.links.model.fpn import bbox_head_loss_pre
from chainercv.links.model.fpn import mask_head_loss_post
from chainercv.links.model.fpn import mask_head_loss_pre
from chainercv.links.model.fpn import rpn_loss


class CascadeRCNNTrainChain(chainer.Chain):

    def __init__(self, model):
        super(CascadeRCNNTrainChain, self).__init__()
        with self.init_scope():
            self.model = model

    def forward(self, imgs, bboxes, labels, scales):
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
            'loss': 0, 'loss/bbox_head/loc': 0, 'loss/bbox_head/conf': 0
        }
        for i, bbox_head in enumerate(self.model.bbox_heads):
            rois, roi_indices, head_gt_locs, head_gt_labels = \
                bbox_head_loss_pre(
                    rois, roi_indices, bbox_head.std, bboxes, labels)
            head_locs, head_confs = bbox_head(hs, rois, roi_indices)
            head_loc_loss, head_conf_loss = bbox_head_loss_post(
                head_locs, head_confs,
                roi_indices, head_gt_locs, head_gt_labels, B)
            rois = bbox_head.decode_bbox(
                rois, roi_indices, head_locs, scales, sizes)

            loss += (rpn_loc_loss + rpn_conf_loss +
                     head_loc_loss + head_conf_loss)
        chainer.reporter.report({
            'loss': loss,
            'loss/rpn/loc': rpn_loc_loss, 'loss/rpn/conf': rpn_conf_loss,
            'loss/bbox_head/loc': head_loc_loss,
            'loss/bbox_head/conf': head_conf_loss},
            self)
        return loss
