from transforms import Compose, Flip, Normalize, Sacle


def setup_transform(cfg, mean):
    transforms = Compose()
    transforms.append(Flip())
    transforms.append(Sacle(cfg.min_size, cfg.max_size))
    transforms.append(Normalize(mean))
    return transforms
