"""
A set of helpers for the examples
"""
import argparse


def base_parser():
    """
    Create a parser with the base parameters

    :return: argparse.ArgumentParser instance
    """
    parser = argparse.ArgumentParser(description="Train a FCNN model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.00001, help="Learning rate")
    parser.add_argument("--nb_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--model_dir", required=True, help="Path to save model")
    return parser


def check_files_order(files1, files2):
    """
    Here we check that the two input lists of str are correctly sorted.
    Except for the last, splits of files1[i] and files2[i] from the "_" character, must be equal.

    :param files1: list of filenames (str)
    :param files2: list of filenames (str)
    """
    assert len(files1) == len(files2)

    def get_basename(n):
        return "_".join([n.split("_")][:-1])

    for p, l in zip(files1, files2):
        assert get_basename(p) == get_basename(l)
