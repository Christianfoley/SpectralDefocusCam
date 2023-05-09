# --------------------------------------------------#
#               Spectral DefocusCam                #
#                                                  #
#   Use to run train, inference, or evaluation     #
#   modules by refering to an appropriate config   #
#   file. For examples see:                        #
#       config_files/TEMPLATE_*.yml                #
# --------------------------------------------------#

import argparse

import train
import inference
import evaluation


def parse_args():
    """
    Parse command line arguments
    In python namespaces are implemented as dictionaries

    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module",
        type=str,
        help="name of module to run. Either 'train', 'inference', 'evaluate",
    )

    parser.add_argument(
        "--config",
        type=str,
        help="path to yaml configuration file",
    )
    args = parser.parse_args()
    return args


def main(module, config):
    """
    Run the specified module with the given configuration.
    """
    if module == "train":
        train.main(config)
    if module == "inference":
        inference.main(config)
    if module == "evaluation":
        evaluation.main(config)


if __name__ == "__main__":
    args = parse_args()
    main(args.module, args.config)
