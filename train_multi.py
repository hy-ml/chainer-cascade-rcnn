import argparse
import multiprocessing
import numpy as np
import cProfile
import io
import pstats

import chainer
from chainer import serializers
from chainer import training
import chainermn
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
    parser.add_argument('--benchmark_n_iteration', type=int, default=50,
                        help='Iteration in benchmark option. Default is 50.')
    parser.add_argument('--n_print_profile', type=int, default=100,
                        help='Default is 100.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg.merge_from_file(args.config)
    cfg.path = args.config
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

    comm = chainermn.create_communicator('pure_nccl')
    assert comm.size == cfg.n_gpu
    device = comm.intra_rank

    if comm.rank == 0:
        print(cfg)

    model = setup_model(cfg)
    train_chain = CascadeRCNNTrainChain(model)
    chainer.cuda.get_device_from_id(device).use()
    train_chain.to_gpu()

    # TODO: refactor not to split datasets
    dataset = setup_dataset(cfg, 'train')
    # transform = setup_transform(cfg, model.extractor.mean)
    # train_dataset = dataset.slice[:, ('img', 'bbox', 'label')]
    # train_dataset = TransformDataset(train_dataset, ('img', 'bbox', 'label'),
    #                                  transform)

    if args.benchmark:
        shuffle = False
    else:
        shuffle = True

    if comm.rank == 0:
        # indices = np.arange(len(train_dataset))
        indices = np.arange(len(dataset))
    else:
        indices = None

    indices = chainermn.scatter_dataset(indices, comm, shuffle=shuffle)
    # train_dataset = train_dataset.slice[indices]
    dataset = dataset.slice[indices]

    transform = setup_transform(cfg, model.extractor.mean)
    train_dataset = dataset.slice[:, ('img', 'bbox', 'label')]
    train_dataset = TransformDataset(train_dataset, ('img', 'bbox', 'label'),
                                     transform)

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
    optimizer = chainermn.create_multi_node_optimizer(
        setup_optimizer(cfg), comm)
    optimizer = optimizer.setup(train_chain)
    optimizer = add_hook_optimizer(optimizer, cfg)
    freeze_params(cfg, train_chain.model)

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=device, converter=converter)
    if args.benchmark:
        stop_trigger = (args.benchmark_n_iteration, 'iteration')
        outdir = 'benchmark_out'
    else:
        stop_trigger = (cfg.solver.n_iteration, 'iteration')
        outdir = get_outdir(args.config)
    trainer = training.Trainer(updater, stop_trigger, outdir)

    if args.benchmark:
        if comm.rank == 0:
            log_interval = 10, 'iteration'
            trainer.extend(training.extensions.LogReport(trigger=log_interval))
            trainer.extend(training.extensions.PrintReport(
                ['epoch', 'iteration', 'main/loss',
                 'main/loss/bbox_head/loc', 'main/bbox_head/conf']),
                trigger=log_interval)
        pr = cProfile.Profile()
        pr.enable()
        trainer.run()
        pr.disable()
        s = io.StringIO()
        sort_by = 'tottime'
        ps = pstats.Stats(pr, stream=s).sort_stats(sort_by)
        ps.print_stats()
        if comm.rank == 0:
            lines = s.getvalue().split('\n')
            for line in lines[:args.n_print_profile]:
                print(line)

        pr.dump_stats(
            '{0}/train_multi_rank_{1}.cprofile'.format(outdir, comm.rank))
        exit()

    setup_extension(cfg, trainer, model, comm)

    if args.resume:
        serializers.load_npz(args.resume, trainer, strict=False)

    trainer.run()


if __name__ == '__main__':
    main()
