from models import CascadeResNet50, CascadeRCNNResNet101


def setup_model(cfg):
    n_fg_class = cfg.dataset.n_fg_class
    pretrained_model = cfg.model.pretrained_model
    min_size = cfg.model.min_size
    max_size = cfg.model.max_size

    if cfg.model.type == 'CascadeRCNNResNet50':
        model = CascadeResNet50(n_fg_class, pretrained_model,
                                min_size=min_size, max_size=max_size)
    elif cfg.model.type == 'CascadeRCNNResNet101':
        model = CascadeRCNNResNet101(n_fg_class, pretrained_model,
                                     min_size=min_size, max_size=max_size)
    else:
        raise ValueError('Not support model `{}`.'.format(cfg.model.type))
    return model


def freeze_params(cfg, model):
    for path, link in model.namedlinks():
        for regex in cfg.model.freeze_param:
            if re.fullmatch(regex, path):
                link.disable_update()
                break
