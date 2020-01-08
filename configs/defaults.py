from yacs.config import CfgNode as CN


_C = CN()

# model
_C.model = CN()
_C.model.type = ''
_C.model.pretrained_model = 'imagenet'
_C.model.freeze_param = [
    '/.+/bn',
    '/extractor/base/conv1',
    '/extractor/base/res2'
]

# dataset
_C.dataset = CN()
_C.dataset.train = ''
_C.dataset.eval = ''
_C.dataset.n_fg_class = 0

# solver
_C.solver = CN()
_C.solver.optimizer = 'MomentumSGD'
_C.solver.base_lr = 0.02
_C.solver.weight_decay = 0.0001
_C.solver.momentum = 0.9
_C.solver.hooks = ['WeightDecay']
_C.solver.n_iteration = 90000
_C.solver.lr_gamma = 0.1
_C.solver.lr_step = [60000, 80000]
_C.solver.lr_warm_up_duration = 500
_C.solver.lr_warm_up_rate = 1 / 3
_C.solver.gradient_clipping_thresh = 5


# misc
_C.min_size = 800
_C.max_size = 1333
_C.n_gpu = 8
_C.n_sample_per_gpu = 2
_C.n_worker = 4
_C.workspace_size = 512  # MB
_C.autotune = True
_C.cudnn_fast_batch_normalization = True
_C.path = ''  # this value should be overwritten
# FIXME: remove
_C.debug = False
