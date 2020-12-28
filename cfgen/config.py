# coding=utf-8
# Copyright (C) The Zhejiang University DMAC Lab Authors. team - All Rights Reserved
#
# Written by Xiangji Zeng <xijiz@qq.com>, March 2020
"""Configurations"""

import os
from typing import List, Dict, Any
from datetime import datetime
import torch


class TrainerConfig:
    """
    Trainer Configuarion.

    Args:
        model (str): Model name to be trained.
        dataset (str): Dataset name to be trained.
        tokenizer (str): Tokenizer name.
        data_folder (str): The root folder of data.
        gpu (List[int]): The given GPU devices.
        hyperparameters (Dict[str, Any]): All model hyperparameters.
        learning_rate (float): Learning rate.
        batch_size (int): Batch size.
        epoch (int): Training epoch.
        stop_if_no_improvement (int): This parameter is used to decease the training epoch if the metric does not
            increase with a given number of epoch round.
        early_stop_loss (float): Early stop loss for overcoming overfitting.
        identity (str): The sub-folder name of an experiment.
    """

    def __init__(
            self,
            model: str,
            dataset: str,
            tokenizer: str,
            data_folder: str,
            gpu: List[int],
            hyperparameters: Dict[str, Any],
            learning_rate: float,
            batch_size: int,
            epoch: int,
            stop_if_no_improvement: int = 15,
            early_stop_loss: float = 0.01,
            identity: str = None
    ):
        super(TrainerConfig, self).__init__()
        self.model = model
        self.dataset = dataset
        self.tokenizer = tokenizer
        log_folder = os.path.join(data_folder, "logs")
        if identity is not None and os.path.isdir(log_folder) and identity in os.listdir(log_folder):
            self.identity = identity
        else:
            self.identity = "{0}-{1}-{2}".format(dataset, model, datetime.now().strftime("%Y%m%d%H%M%S"))
        self.data_folder = data_folder
        self.output_folder = os.path.join(self.data_folder, "logs", self.identity)
        self.dataset_folder = os.path.join(self.data_folder, "datasets", self.dataset)
        self.tokenizer_folder = os.path.join(self.data_folder, "tokenizers", self.tokenizer)
        self.device, self.gpu = self._devices(gpu)
        self.hyperparameters = hyperparameters
        self.max_seq_len = hyperparameters.get("max_seq_len", 512)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch = epoch
        self.stop_if_no_improvement = stop_if_no_improvement
        self.early_stop_loss = early_stop_loss

    def _devices(self, gpu: List[int]):
        """
        Load available computing devices.

        Args:
            gpu (List[int]): All GPU devices that user gives.
        """
        device, avaiable_gpu = "cpu", []
        if torch.cuda.is_available() and len(gpu) > 0:
            if torch.cuda.device_count() < len(gpu):
                avaiable_gpu = list(range(torch.cuda.device_count()))
            else:
                avaiable_gpu = gpu
            device = "cuda:{0}".format(avaiable_gpu[0])

        return (device, avaiable_gpu)
