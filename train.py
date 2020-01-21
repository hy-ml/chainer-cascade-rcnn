import argparse
import multiprocessing
import numpy as np
import cProfile
import io
import pstats

import chainer
from chainer import serializers
from chainer import training
from chainercv.chainer_experimental.datasets.sliceable import TransformDataset

from configs import cfg
from utils.path import get_outdir
from models import CascadeRCNNTrainChain
from setup_helpers import setup_dataset
from setup_helpers import setup_order_sampler
from setup_helpers import setup_model, freeze_params
from setup_helpers import setup_transform
from setup_helpers import setup_optimizer, add_hook_optimizer
from setup_helpers import setup_extension


def converter(batch, device=None):
    # do not send data to gpu (device is ignored)
    return tuple(list(v) for v in zip(*batch))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str,
                        help='Path to the config file.')
    parser.add_argument('--resume', type=str)
    parser.add_argument('--benchmark', action='store_true',
                        help='Benchmark option.')
    parser.add_argument('--benchmark_n_iteration', type=int, default=500,
                        help='Iteration in benchmark option. Default is 500.')
    parser.add_argument('--n_print_profile', type=int, default=100,
                        help='Default is 100.')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg.merge_from_file(args.config)
    cfg.freeze()

    chainer.cuda.set_max_workspace_size(cfg.workspace_size * 1024 * 1024)
    chainer.config.autotune = cfg.autotune
    chainer.config.cudnn_fast_batch_normalization = \
        cfg.cudnn_fast_batch_normalization

    if hasattr(multiprocessing, 'set_start_method'):
        multiprocessing.set_start_method('forkserver')
        p = multiprocessing.Process()
        p.start()
        p.join()

    print(cfg)

    model = setup_model(cfg)
    train_chain = CascadeRCNNTrainChain(model)
    train_chain.to_gpu(args.gpu)

    transform = setup_transform(cfg, model.extractor.mean)
    dataset = setup_dataset(cfg, 'train')
    transform = setup_transform(cfg, model.extractor.mean)
    train_dataset = dataset.slice[:, ('img', 'bbox', 'label')]
    train_dataset = TransformDataset(train_dataset, ('img', 'bbox', 'label'),
                                     transform)

    if args.benchmark:
        shuffle = False
    else:
        shuffle = True

    order_sampler = setup_order_sampler(cfg, dataset)
    if order_sampler is None:
        shuffle = shuffle
    else:
        shuffle = None

    train_iter = chainer.iterators.MultiprocessIterator(
        train_dataset, cfg.n_sample_per_gpu,
        n_processes=cfg.n_worker,
        shared_mem=100 * 1000 * 1000 * 4, shuffle=shuffle,
        order_sampler=order_sampler)
    # train_iter = chainer.iterators.MultiprocessIterator(
    #     train_dataset, cfg.n_sample_per_gpu,
    #     n_processes=cfg.n_worker,
    #     shared_mem=100 * 1000 * 1000 * 4, shuffle=shuffle)

    optimizer = setup_optimizer(cfg)
    optimizer = optimizer.setup(train_chain)
    optimizer = add_hook_optimizer(optimizer, cfg)
    freeze_params(cfg, train_chain.model)

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, converter=converter)
    if args.benchmark:
        stop_trigger = (args.benchmark_n_iteration, 'iteration')
        outdir = 'benchmark_out'
    else:
        stop_trigger = (cfg.solver.n_iteration, 'iteration')
        outdir = get_outdir(args.config)
    trainer = training.Trainer(updater, stop_trigger, outdir)

    if args.benchmark:
        log_interval = 10, 'iteration'
        trainer.extend(training.extensions.LogReport(trigger=log_interval))
        trainer.extend(training.extensions.PrintReport(
            ['epoch', 'iteration', 'main/loss',
                'main/loss/bbox_head/loc', 'main/loss/bbox_head/conf']),
            trigger=log_interval)
        pr = cProfile.Profile()
        pr.enable()
        trainer.run()
        pr.disable()
        s = io.StringIO()
        sort_by = 'tottime'
        ps = pstats.Stats(pr, stream=s).sort_stats(sort_by)
        ps.print_stats()
        lines = s.getvalue().split('\n')
        for line in lines[:args.n_print_profile]:
            print(line)

        pr.dump_stats(
            '{0}/train.cprofile'.format(outdir))
        exit()

    setup_extension(cfg, trainer, model, None)

    if args.resume:
        serializers.load_npz(args.resume, trainer, strict=False)

    trainer.run()


if __name__ == '__main__':
    main()
