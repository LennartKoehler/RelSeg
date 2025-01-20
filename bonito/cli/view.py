"""
Bonito model viewer - display a model architecture for a given config.
"""

import os.path

import toml
import argparse
from bonito.util import load_symbol
from visualization import visualize


def main(args):
    
    if os.path.isdir(args.config):
        config = toml.load(os.path.join(args.config, "config.toml"))
    else:
        config = toml.load(args.config)
    Model = load_symbol(config, "Model")
    model = Model(config).to("cuda")
    # batchsize = config["basecaller"]["batchsize"]
    batchsize = 1
    chunksize = config["basecaller"]["chunksize"]
    channels = 1
    print(model)
    print("Total parameters in model", sum(p.numel() for p in model.parameters()))
    visualize(model, (batchsize, channels, chunksize))

def test_main(config):
    if os.path.isdir(config):
        config = toml.load(os.path.join(config, "config.toml"))
    else:
        config = toml.load(config)
    Model = load_symbol(config, "Model")
    model = Model(config).to("cuda")
    # batchsize = config["basecaller"]["batchsize"]
    batchsize = 1
    chunksize = config["basecaller"]["chunksize"]
    channels = 1
    print(model)
    print("Total parameters in model", sum(p.numel() for p in model.parameters()))
    visualize(model, (batchsize, channels, chunksize))

def argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("config")
    return parser
