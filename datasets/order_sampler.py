from copy import deepcopy
import numpy as np

from chainer.iterators import OrderSampler


class GroupOrderSampler(OrderSampler):
    def __init__(self, group_indices, batch_size):
        self._group_indices = group_indices
        self._batch_size = batch_size

    def __call__(self, current_order, current_position):
        group_pair_indices = \
            [self._make_pair_indices(gi) for gi in self._group_indices]
        pair_indices = np.vstack(group_pair_indices)
        pair_indices = np.random.permutation(pair_indices)
        order = pair_indices.reshape(-1)
        return order

    def _make_pair_indices(self, indices):
        indices = np.random.permutation(np.array(indices))
        indices = indices[:int(
            indices.shape[0] - indices.shape[0] % self._batch_size)]
        pair_indices = indices.reshape(-1, self._batch_size)
        return pair_indices


class AspectRatioOrderSampler(GroupOrderSampler):
    def __init__(self, dataset, batch_size):
        group_indices = self._split_indices(dataset)
        super(AspectRatioOrderSampler, self).__init__(
            group_indices, batch_size)

    def _split_indices(self, dataset):
        dataset = deepcopy(dataset).slice[:, 'aspect_ratio']
        width_indices = []
        height_indices = []
        for i, aspect_ratio in enumerate(dataset):
            if aspect_ratio < 1:
                width_indices.append(i)
            else:
                height_indices.append(i)
        return width_indices, height_indices
