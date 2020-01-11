from chainer import training

from utils.path import get_logdir
from extensions import LogTensorboard, LrScheduler


def setup_extension(cfg, trainer, model, comm=None):
    if comm is None or comm.rank == 0:
        log_interval = 10, 'iteration'
        trainer.extend(training.extensions.LogReport(trigger=log_interval))
        trainer.extend(training.extensions.observe_lr(), trigger=log_interval)
        trainer.extend(training.extensions.PrintReport(
            ['epoch', 'iteration', 'lr', 'main/loss',
             'main/loss/bbox_head/loc', 'main/loss/bbox_head/conf']),
            trigger=log_interval)
        trainer.extend(training.extensions.ProgressBar(update_interval=10))

        trainer.extend(training.extensions.snapshot(),
                       trigger=(10000, 'iteration'))
        trainer.extend(
            training.extensions.snapshot_object(
                model, 'model_iter_{.updater.iteration}'),
            trigger=(cfg.solver.n_iteration, 'iteration'))
        trainer.extend(
            LogTensorboard([
                'lr', 'main/loss',
                'main/loss/bbox_head/loc',
                'main/loss/bbox_head/conf',
                'main/loss/bbox_head/stage0/loc',
                'main/loss/bbox_head/stage0/conf',
                'main/loss/bbox_head/stage1/loc',
                'main/loss/bbox_head/stage1/conf',
                'main/loss/bbox_head/stage2/loc',
                'main/loss/bbox_head/stage2/conf'],
                trigger=(10, 'iteration'), log_dir=get_logdir(cfg.path)))

    trainer.extend(LrScheduler(
        cfg.solver.base_lr, cfg.solver.lr_gamma, cfg.solver.lr_step,
        cfg.solver.lr_warm_up_duration, cfg.solver.lr_warm_up_rate))
