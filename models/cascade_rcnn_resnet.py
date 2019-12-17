from __future__ import division

import chainer
import chainer.functions as F
import chainer.links as L

from chainercv.links.model.resnet import ResNet101
from chainercv.links.model.resnet import ResNet50
from chainercv import utils

from models.bbox_head import BboxHead
from models.cascade_rcnn import CascadeRCNN
from models.fpn import FPN
from models.rpn import RPN


class CascadeRCNNResNet(CascadeRCNN):
    """Base class for Faster R-CNN with a ResNet backbone and FPN.
    A subclass of this class should have :obj:`_base` and :obj:`_models`.
    Args:
        n_fg_class (int): The number of classes excluding the background.
        pretrained_model (string): The weight file to be loaded.
            This can take :obj:`'coco'`, `filepath` or :obj:`None`.
            The default value is :obj:`None`.
            * :obj:`'coco'`: Load weights trained on train split of \
                MS COCO 2017. \
                The weight file is downloaded and cached automatically. \
                :obj:`n_fg_class` must be :obj:`80` or :obj:`None`.
            * :obj:`'imagenet'`: Load weights of ResNet-50 trained on \
                ImageNet. \
                The weight file is downloaded and cached automatically. \
                This option initializes weights partially and the rests are \
                initialized randomly. In this case, :obj:`n_fg_class` \
                can be set to any number.
            * `filepath`: A path of npz file. In this case, :obj:`n_fg_class` \
                must be specified properly.
            * :obj:`None`: Do not load weights.
        return_values (list of strings): Determines the values
            returned by :meth:`predict`.
        min_size (int): A preprocessing paramter for :meth:`prepare`. Please \
            refer to :meth:`prepare`.
        max_size (int): A preprocessing paramter for :meth:`prepare`.
    """

    _stds = [(0.1, 0.2), (0.05, 0.1), (0.033, 0.067)]

    def __init__(self, n_fg_class=None, pretrained_model=None,
                 return_values=['bboxes', 'labels', 'scores'],
                 min_size=800, max_size=1333):
        param, path = utils.prepare_pretrained_model(
            {'n_fg_class': n_fg_class}, pretrained_model, self._models)

        base = self._base(n_class=1, arch='he')
        base.pick = ('res2', 'res3', 'res4', 'res5')
        base.pool1 = lambda x: F.max_pooling_2d(
            x, 3, stride=2, pad=1, cover_all=False)
        base.remove_unused()
        extractor = FPN(
            base, len(base.pick), (1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64))
        bbox_heads = chainer.ChainList(
            [BboxHead(param['n_fg_class'] + 1, extractor.scales, std)
             for std in self._stds])

        super(CascadeRCNNResNet, self).__init__(
            extractor=extractor,
            rpn=RPN(extractor.scales),
            bbox_heads=bbox_heads,
            return_values=return_values,
            min_size=min_size, max_size=max_size
        )

        if path == 'imagenet':
            _copyparams(
                self.extractor.base,
                self._base(pretrained_model='imagenet', arch='he'))
        elif path:
            chainer.serializers.load_npz(path, self)


class CascadeResNet50(CascadeRCNNResNet):
    """Faster R-CNN with ResNet-50 and FPN.
    Please refer to :class:`~chainercv.links.model.fpn.FasterRCNNFPNResNet`.
    """

    _base = ResNet50
    _models = {
        'coco': {
            'param': {'n_fg_class': 80},
            'url': 'https://chainercv-models.preferred.jp/'
            'faster_rcnn_fpn_resnet50_coco_trained_2019_03_15.npz',
            'cv2': True
        },
    }


class CascadeRCNNResNet101(CascadeRCNNResNet):
    """Faster R-CNN with ResNet-101 and FPN.
    Please refer to :class:`~chainercv.links.model.fpn.FasterRCNNFPNResNet`.
    """

    _base = ResNet101
    _models = {
        'coco': {
            'param': {'n_fg_class': 80},
            'url': 'https://chainercv-models.preferred.jp/'
            'faster_rcnn_fpn_resnet101_coco_trained_2019_03_15.npz',
            'cv2': True
        },
    }


def _copyparams(dst, src):
    if isinstance(dst, chainer.Chain):
        for link in dst.children():
            _copyparams(link, src[link.name])
    elif isinstance(dst, chainer.ChainList):
        for i, link in enumerate(dst):
            _copyparams(link, src[i])
    else:
        dst.copyparams(src)
        if isinstance(dst, L.BatchNormalization):
            dst.avg_mean = src.avg_mean
            dst.avg_var = src.avg_var
