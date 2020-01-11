from chainer.training import extension


# TODO: support option not to use warm up
class LrScheduler(extension.Extension):
    def __init__(self, base_lr, gamma, step, warm_up_duration, warm_up_rate):
        self._base_lr = base_lr
        self._gamma = gamma
        self._step = step
        self._warm_up_duration = warm_up_duration
        self._warm_up_rate = warm_up_rate

    def __call__(self, trainer):
        iteration = trainer.updater.iteration
        if iteration < self._warm_up_duration:
            rate = self._warm_up_rate \
                + (1 - self._warm_up_rate) * iteration / self._warm_up_duration
        else:
            rate = 1
            for step in self._step:
                if iteration >= step:
                    rate *= self._gamma

        lr = self._base_lr * rate
        self._update_value(trainer, lr)
        return lr

    def _update_value(self, trainer, value):
        optimizer = trainer.updater.get_optimizer('main')
        setattr(optimizer, 'lr', value)
        self._last_value = value
