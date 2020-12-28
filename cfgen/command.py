# coding=utf-8
# Copyright (C) The Zhejiang University DMAC Lab Authors. team - All Rights Reserved
#
# Written by Xiangji Zeng <xijiz@qq.com>, March 2020
"""Command Line Tools"""

from typing import List
import fire
import torch
from .laboratory import train, trainall
from .visualization import visualize_avearage_causal_effects


torch.manual_seed(999)


def check(model_names: List[str], dataset_names: List[str], n_seeds: List[int]) -> None:
    """
    Check whether the input is legal.

    Args:
        model_names (List[str]): The inputted model names by user.
        dataset_names (List[str]): The inputted dataset names by user.
        n_seeds (List[int]): The inputted training seeds by user.
    """
    all_model_names, all_dataset_names = ["bilstm", "bert"], ["cluener", "cner"]
    if not all([model_name in all_model_names for model_name in model_names]):
        print("Available models {0}, but with provided models {1}".format(all_model_names, model_names))
        exit(0)
    if not all([dataset_name in all_dataset_names for dataset_name in dataset_names]):
        print("Available datasets {0}, but with provided datasets {1}".format(all_dataset_names, dataset_names))
        exit(0)
    if not all([seed <= 500 or seed == -1 for seed in n_seeds]):
        print("All seeds must be less equal to 500 or -1.")
        exit(0)


class CommandLineTool:
    """
    CommandLineTool provides interfaces to do experiments with various settings.
    """

    @staticmethod
    def trainall(
            models: List[str] = ["bilstm", "bert"],
            datasets: List[str] = ["cluener", "cner"],
            seeds: List[int] = [100, 200, 300, 400, 500, -1],
            gpu: List[int] = [],
            batch_size: int = 8,
            epoch: int = 128,
            data_folder: str = "./tmp/"
    ):
        """
        Train all models on all datasets with all settings.

        Args:
            models (List[str]): The inputted model names by user.
            datasets (List[str]): The inputted dataset names by user.
            seeds (List[int]): The inputted training seeds by user.
            gpu (List[int]): A list of GPU devices.
            batch_size (int): Training batch size.
            epoch (int): The total training epoch.
            data_folder (str): The root folder where data exists.
        """
        check(models, datasets, seeds)
        trainall(models, datasets, seeds, gpu, batch_size, epoch, data_folder)

    @staticmethod
    def train(model: str, dataset: str, seed: int, gpu: List[int], batch_size: int = 8, epoch: int = 128, data_folder: str = "./tmp/"):
        """
        Train a model on a dataset with a given seed.

        Args:
            model (str): The inputted model name by user.
            dataset (str): The inputted dataset name by user.
            seed (int): The inputted training seed by user.
            gpu (List[int]): A list of GPU devices.
            batch_size (int): Training batch size.
            epoch (int): The total training epoch.
            data_folder (str): The root folder where data exists.
        """
        check([model], [dataset], [seed])
        results = train(model, dataset, seed, gpu, batch_size, epoch, data_folder)
        print(results)

    @staticmethod
    def visualize(logfolder: str = "./tmp/"):
        """
        Visualize ACE under with the augmented data and the non-augmented data.

        Args:
            logfolder (str): The folder where the log file exists.
        """
        visualize_avearage_causal_effects(logfolder)


def command():
    """Start a CLI."""
    fire.Fire(CommandLineTool)
