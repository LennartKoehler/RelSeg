from util import *
from util import __models_dir__
from util import _load_model

def load_model_lrp(dirname, device, weights=None, half=None, chunksize=None, batchsize=None, overlap=None, quantize=False, use_koi=False):
    """
    Load a model config and weights off disk from `dirname`.
    """
    if not os.path.isdir(dirname) and os.path.isdir(os.path.join(__models_dir__, dirname)):
        dirname = os.path.join(__models_dir__, dirname)
    weights = get_last_checkpoint(dirname) if weights is None else os.path.join(dirname, 'weights_%s.tar' % weights)
    config = toml.load(os.path.join(dirname, 'config.toml'))
    config = set_config_defaults(config, chunksize, batchsize, overlap, quantize)
    config["model"]["package"] = "bonito.lrp.transformer" # this is the change for lrp, because the location is usaully in the .toml file, changing this would break everything
    return _load_model(weights, config, device, half, use_koi)
