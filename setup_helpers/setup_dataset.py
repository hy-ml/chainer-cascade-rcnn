# from chainer.datasets import ConcatenatedDataset
from chainercv.chainer_experimental.datasets.sliceable import ConcatenatedDataset

from datasets import COCOBboxDataset, VOCBboxDataset
from datasets import AspectRatioOrderSampler


def setup_dataset(cfg, split):
    if split == 'train':
        dataset_type = cfg.dataset.train
    elif split == 'eval':
        dataset_type = cfg.dataset.eval
    else:
        raise ValueError()

    # FIXME: remove
    if cfg.debug:
        dataset = COCOBboxDataset(split='debug', year='2017')
        return dataset

    if dataset_type == 'COCO':
        if split == 'train':
            dataset = COCOBboxDataset(split='train', year='2017')
        elif split == 'eval':
            dataset = COCOBboxDataset(
                split='val', year='2017', use_crowded=True,
                return_area=True, return_crowded=True)
        else:
            raise ValueError()
    elif dataset_type == 'VOC':
        if split == 'train':
            dataset = ConcatenatedDataset(
                VOCBboxDataset(year='2007', split='trainval'),
                VOCBboxDataset(year='2012', split='trainval')
            )
        elif split == 'eval':
            dataset = VOCBboxDataset(
                split='test', year='2007', use_difficult=True,
                return_difficult=True)
        else:
            raise ValueError()
    else:
        raise ValueError()

    return dataset


def setup_order_sampler(cfg, dataset=None):
    if cfg.dataset.order_sampler == 'AspectRatioOrderSampler':
        assert dataset is not None
        bs = cfg.n_sample_per_gpu
        order_sampler = AspectRatioOrderSampler(dataset, bs)
    # use default order sampler: ShuffleOrderSampler
    elif cfg.dataset.order_sampler == '':
        order_sampler = None
    else:
        raise ValueError('Not support order sampler: {}.'.format(
            cfg.dataset.order_sampler))

    return order_sampler
