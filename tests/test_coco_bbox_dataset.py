import argparse
import numpy as np

import _init_path  # NOQA
from datasets import COCOBboxDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val'])
    args = parser.parse_args()
    return args


def test_coco_bbox_dataset():
    args = parse_args()

    dataset = COCOBboxDataset(split=args.split)
    for i, data in enumerate(dataset):
        img, _, _, aspect_ratio = data
        _, h, w = img.shape
        assert np.round(w / h, 2) == np.round(aspect_ratio, 2)
        if i > 100:
            break
    print('Success.')


if __name__ == '__main__':
    test_coco_bbox_dataset()
