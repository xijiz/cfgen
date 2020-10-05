# coding=utf-8
# Copyright (C) The Zhejiang University DMAC Lab Authors. team - All Rights Reserved
#
# Written by Xiangji Zeng <xijiz@qq.com>, March 2020
"""Laboratory"""

import os
from typing import List, Any
import pandas as pd
from .config import TrainerConfig
from .trainer import NERTrainer, __DATASET_MAP__
from .common import save_jsonl


__HPS_MAP__ = {"bilstm": {"hidden_dim": 512, "n_layers": 2, "dropout": 0.1}, "bert": {}}
__LR_MAP__ = {"bilstm": 0.001, "bert": 0.00002}
__TOKENIZER_NAME_MAP__ = {"bilstm": "tce", "bert": "bert"}


def trainall(model_names: List[str], dataset_names: List[str], n_seeds: List[int], gpu: List[int], batch_size: int, epoch: int, data_folder: str) -> None:
    """
    Train all models on all datasets with all settings.

    Args:
        model_names (List[str]): The inputted model names by user.
        dataset_names (List[str]): The inputted dataset names by user.
        n_seeds (List[int]): The inputted training seeds by user.
        gpu (List[int]): A list of GPU devices.
        batch_size (int): Training batch size.
        epoch (int): The total training epoch.
        data_folder (str): The root folder where data exists.
    """
    records_file_path = os.path.join(data_folder, "experiments.csv")
    if not os.path.isfile(records_file_path):
        records = pd.DataFrame(columns=["dataset", "model", "n_seed", "f1", "p", "r", "f1-aug", "p-aug", "r-aug", "diff-f1"])
    else:
        records = pd.read_csv(records_file_path)
    for dataset_name in dataset_names:
        for model_name in model_names:
            for seed in n_seeds:
                if len(records.query("dataset == '{0}' & model == '{1}' & n_seed == '{2}'".format(dataset_name, model_name, seed))) > 0:
                    continue
                results = train(model_name, dataset_name, seed, gpu, batch_size, epoch, data_folder)
                records.loc[len(records)] = results
                records.to_csv(records_file_path, index=False)
                print("Experimental data has been saved into {records_file_path}".format(records_file_path=records_file_path))


def train(model_name: str, dataset_name: str, seed: int, gpu: List[int], batch_size: int, epoch: int, data_folder: str) -> List[Any]:
    """
    Train a model on a dataset with a given seed.

    Args:
        model_name (str): The inputted model name by user.
        dataset_name (str): The inputted dataset name by user.
        seed (int): The inputted training seed by user.
        gpu (List[int]): A list of GPU devices.
        batch_size (int): Training batch size.
        epoch (int): The total training epoch.
        data_folder (str): The root folder where data exists.
    """
    # train with observational examples
    config = TrainerConfig(
        model_name, dataset_name, __TOKENIZER_NAME_MAP__[model_name], data_folder,
        gpu, __HPS_MAP__[model_name], __LR_MAP__[model_name], batch_size, epoch
    )
    datasets = list(__DATASET_MAP__[dataset_name].split_datasets(config.dataset_folder, config.max_seq_len))
    n_trains = len(datasets[0])
    if seed == -1:
        seed = n_trains
    datasets[0] = datasets[0][:seed]  # extract partial obervational examples
    trainer = NERTrainer(config, datasets)
    results1 = trainer.train()
    # train with counterfactual examples
    if seed != n_trains:
        reasonable_cfexamples, unreasonable_cfexamples = trainer.create_discriminated_examples(datasets[0][:seed])
        datasets[0] += reasonable_cfexamples
        del trainer
        trainer = NERTrainer(config, datasets)
        results2 = trainer.train()
        del trainer
        testset = datasets[2]
        for i, _ in enumerate(testset):
            testset[i]["nsp"] = {"noaug": results1["nsp"][i], "aug": results2["nsp"][i]}
        save_jsonl(reasonable_cfexamples, config.output_folder, "train-{0}-reasonable-{1}.jsonl".format(model_name, seed))
        save_jsonl(unreasonable_cfexamples, config.output_folder, "train-{0}-unreasonable-{1}.jsonl".format(model_name, seed))
        save_jsonl(testset, config.output_folder, "test-{0}-{1}.jsonl".format(model_name, seed))
    else:
        results2 = results1
    merged_results = [
        dataset_name, model_name, seed,
        results1["entity"]["f1"], results1["entity"]["p"], results1["entity"]["r"],
        results2["entity"]["f1"], results2["entity"]["p"], results2["entity"]["r"],
        results2["entity"]["f1"] - results1["entity"]["f1"]
    ]

    return merged_results
