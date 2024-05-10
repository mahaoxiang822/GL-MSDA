from mmcv.utils import Registry, build_from_cfg

TRANSFORMER = Registry('Transformer')
POSITIONAL_ENCODING = Registry('Position encoding')
OBJECTPAIRPOOLLAYER = Registry('ObjectPairPoolLayer')

def build_transformer(cfg, default_args=None):
    """Builder for Transformer."""
    return build_from_cfg(cfg, TRANSFORMER, default_args)


def build_positional_encoding(cfg, default_args=None):
    """Builder for Position Encoding."""
    return build_from_cfg(cfg, POSITIONAL_ENCODING, default_args)

def build_object_pair_pool_layer(cfg, default_args=None):
    return build_from_cfg(cfg, OBJECTPAIRPOOLLAYER, default_args)