import argparse
import numpy as np
from chainer.iterators import SerialIterator

import _init_path  # NOQA
from datasets import COCOBboxDataset, VOCBboxDataset
from datasets import AspectRatioOrderSampler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco',
                        choices=['coco', 'voc'])
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val'])
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()
    return args


def test_aspect_ratio_sampler():
    args = parse_args()

    if args.dataset == 'coco':
        dataset = COCOBboxDataset(split=args.split)
    elif args.dataset == 'voc':
        dataset = VOCBboxDataset(split=args.split)
    else:
        raise ValueError()

    order_sampler = AspectRatioOrderSampler(dataset, args.batch_size)

    iterator = SerialIterator(dataset, args.batch_size,
                              False, order_sampler=order_sampler)

    for i, data in enumerate(iterator):
        if (i + 1) % 10 == 0:
            print('{}/{}'.format(i + 1, int(len(dataset) / args.batch_size)))
        aspect_raios = np.empty(len(data))
        for j, d in enumerate(data):
            img = d[0]
            _, h, w = img.shape
            aspect_raios[j] = w / h

        assert np.all(aspect_raios >= 1) or np.all(aspect_raios <= 1)
    print('Success.')


if __name__ == '__main__':
    test_aspect_ratio_sampler()
