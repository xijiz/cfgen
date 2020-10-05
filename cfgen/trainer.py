# coding=utf-8
# Copyright (C) The Zhejiang University DMAC Lab Authors. team - All Rights Reserved
#
# Written by Xiangji Zeng <xijiz@qq.com>, March 2020
"""Trainer"""

import logging
import sys
import os
import math
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .tokenizer import TCETokenizer, BERTTokenizer
from .inadapter import InCNER, InCLUNER
from .outadapter import OutAdapter
from .model import LSTMTagger, BERTTagger
from .config import TrainerConfig
from .metric import evalner
from .generator import create_counterfactual_examples
from .common import collate_fn, to_entities


__TOKENIZER_MAP__ = {"tce": TCETokenizer, "bert": BERTTokenizer}
__MODEL_MAP__ = {"bilstm": LSTMTagger, "bert": BERTTagger}
__DATASET_MAP__ = {"cner": InCNER, "cluener": InCLUNER}


class NERTrainer:
    """
    NERTrainer manages to train and test NER models on different datasets.

    Args:
        config (TrainerConfig): Trainer configuration.
        datasets (Tuple[list, list, list]): Train/Dev/Test set.
    """

    def __init__(self, config: TrainerConfig, datasets: Tuple[list, list, list]):
        super(NERTrainer, self).__init__()
        writer_folder = os.path.join(config.output_folder, "summary")
        if not os.path.isdir(writer_folder):
            os.makedirs(writer_folder)
        self._config = config
        self._tokenizer = __TOKENIZER_MAP__[config.tokenizer](config.tokenizer_folder)
        self._tokenizer.save(config.output_folder)
        self._outadapter = OutAdapter(config.dataset_folder)
        self._outadapter.save(config.output_folder)
        self._trainset = __DATASET_MAP__[config.dataset](datasets[0], config.max_seq_len, self._outadapter, self._tokenizer)
        self._devset = __DATASET_MAP__[config.dataset](datasets[1], config.max_seq_len, self._outadapter, self._tokenizer)
        self._testset = __DATASET_MAP__[config.dataset](datasets[2], config.max_seq_len, self._outadapter, self._tokenizer)
        self._collate_fn = collate_fn(self._tokenizer.pad_id, self._outadapter.pad_id, config.device)
        self._trainloader = DataLoader(self._trainset, config.batch_size, collate_fn=self._collate_fn)
        self._devloader = DataLoader(self._devset, config.batch_size, collate_fn=self._collate_fn)
        self._testloader = DataLoader(self._testset, config.batch_size, collate_fn=self._collate_fn)
        config.hyperparameters["n_tags"] = len(self._outadapter)
        config.hyperparameters["empty_id"] = self._tokenizer.empty_id
        config.hyperparameters.update(self._tokenizer.configs())
        self._model = __MODEL_MAP__[config.model](
            **config.hyperparameters, token_embeddings=self._tokenizer.token_embeddings()
        ).to(config.device)
        if len(config.gpu) > 1:
            self._model = DataParallel(self._model, device_ids=config.gpu)
        self._loss_fn = nn.CrossEntropyLoss()
        self._optimizer = optim.Adam(self._model.parameters(), lr=config.learning_rate)
        self._writer = SummaryWriter(writer_folder)
        formatter = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
        self._logger = logging.getLogger(__name__)
        self._logger.handlers.clear()
        fh = logging.FileHandler(os.path.join(config.output_folder, "log.txt"))
        fh.setFormatter(formatter)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        self._logger.addHandler(fh)
        self._logger.addHandler(sh)
        self._logger.setLevel(logging.DEBUG)

    def load_checkpoints(self) -> None:
        """Load trained checkpoints from local disk."""
        checkpoints_path = os.path.join(self._config.output_folder, "model.checkpoints")
        if os.path.isfile(checkpoints_path):
            checkpoints = torch.load(checkpoints_path, map_location=torch.device("cpu"))
            if isinstance(self._model, DataParallel):
                self._model.module.load_state_dict(checkpoints)
            elif isinstance(self._model, nn.Module):
                self._model.load_state_dict(checkpoints)

    def save_checkpoints(self) -> None:
        """Save trained checkpoints into local disk."""
        checkpoints_path = os.path.join(self._config.output_folder, "model.checkpoints")
        if isinstance(self._model, DataParallel):
            checkpoints = self._model.module.state_dict()
        else:
            checkpoints = self._model.state_dict()
        torch.save(checkpoints, checkpoints_path)

    def log(self, content: str, y_value: float = None, x_value: float = None) -> None:
        """
        Record status by logging, tensorboard. If x_value or y_value is None, we regard content
        as text and record it. Otherwise, we regard content as a record classification for merge
        the same values (x, y). The commonly used record classifications are F1 score, precision
        score, etc.

        Args:
            content (str): Logging content.
            y_value (float): Y input.
            x_value (float): X input.
        """
        if y_value is not None and x_value is not None:
            self._writer.add_scalar("{0}/{1}".format(self._config.identity, content), y_value, x_value)
        else:
            self._writer.add_text("{0}/log".format(self._config.identity), content)
            content = "[{0}-{1}] ".format(self._config.model, self._config.dataset) + content
            self._logger.debug(content)

    def test(self, loader: DataLoader = None) -> Dict[str, Any]:
        """
        Return model's performance.

        Args:
            loader (DataLoader): Dataloader to be tested.
        """
        if loader is None:
            loader = self._testloader
        self._model.eval()
        batch_gold_labels, batch_pred_labels = [], []
        for _, batch in enumerate(loader):
            input_ids, output_ids, masks = batch
            preds_ = self._model(input_ids, masks)
            pred_labels = [[self._outadapter[label_id] for label_id in label_ids] for label_ids in preds_.argmax(dim=-1).tolist()]
            batch_pred_labels.extend(pred_labels)
            gold_labels = [[self._outadapter[label_id] for label_id in label_ids] for label_ids in output_ids.tolist()]
            batch_gold_labels.extend(gold_labels)
            del batch, input_ids, preds_

        return evalner(batch_gold_labels, batch_pred_labels)

    def train(self) -> Dict[str, Any]:
        """Start to train model on a given dataset."""
        no_improvemnet, max_f1_score = 0, -math.inf
        for epoch in range(self._config.epoch):
            self._model.train()
            total_loss = 0.
            for _, batch in enumerate(self._trainloader):
                input_ids, output_ids, masks = batch
                self._model.zero_grad()
                preds_ = self._model(input_ids, masks)
                loss = self._loss_fn(preds_.view(-1, len(self._outadapter)), output_ids.view(-1))
                loss.backward()
                self._optimizer.step()
                total_loss += loss.item()
                del batch, loss, input_ids, output_ids
            train_loss = total_loss / len(self._trainloader)
            evaluations = self.test(self._devloader)
            self.log("f1", evaluations["entity"]["f1"], epoch)
            self.log("p", evaluations["entity"]["p"], epoch)
            self.log("r", evaluations["entity"]["r"], epoch)
            self.log("train_loss", train_loss, epoch)
            self.log("epoch {0} dev-f1: {1}, dev-p: {2}, dev-r: {3}, train-loss: {4}".format(
                epoch, evaluations["entity"]["f1"], evaluations["entity"]["p"], evaluations["entity"]["r"], train_loss
            ))
            if evaluations["entity"]["f1"] > max_f1_score:
                max_f1_score = evaluations["entity"]["f1"]
                no_improvemnet = 0
                self.save_checkpoints()
            else:
                no_improvemnet += 1
            if train_loss < self._config.early_stop_loss or no_improvemnet > self._config.stop_if_no_improvement:
                break
        self.load_checkpoints()
        evaluations = self.test(self._testloader)
        self.log("test-f1: {0}, test-p: {1}, test-r: {2}".format(evaluations["entity"]["f1"], evaluations["entity"]["p"], evaluations["entity"]["r"]))

        return evaluations

    def create_discriminated_examples(self, trainset: List[dict]) -> List[dict]:
        """
        Create new counterfactual examples with a discriminator from a observational seed examples.

        Args:
            trainset (List[dict]): A list of observational seed examples.
        """
        self.load_checkpoints()
        self._model.eval()
        all_cfexamples = create_counterfactual_examples(trainset)
        reasonable_cfexamples, unreasonable_cfexamples = [], []
        dataset = __DATASET_MAP__[self._config.dataset](all_cfexamples, self._config.max_seq_len, self._outadapter, self._tokenizer)
        dataloader = DataLoader(dataset, self._config.batch_size, collate_fn=self._collate_fn)
        for i, batch in enumerate(dataloader):
            input_ids, output_ids, masks = batch
            preds_ = self._model(input_ids, masks)
            pred_labels = [[self._outadapter[label_id] for label_id in label_ids] for label_ids in preds_.argmax(dim=-1).tolist()]
            for j, labels in enumerate(pred_labels):
                text = all_cfexamples[i*self._config.batch_size + j]["text"]
                replaced_spans = all_cfexamples[i*self._config.batch_size + j]["replaced"]
                predicted_spans = ["[{0}]({1}, {2})".format(span["label"], span["start"], span["end"]) for span in to_entities(text, labels)]
                if len(set(replaced_spans).intersection(set(predicted_spans))) == len(replaced_spans):
                    reasonable_cfexamples.append(all_cfexamples[i*self._config.batch_size + j])
                else:
                    unreasonable_cfexamples.append(all_cfexamples[i*self._config.batch_size + j])

        return (reasonable_cfexamples, unreasonable_cfexamples)
