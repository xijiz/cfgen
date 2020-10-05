# coding=utf-8
# Copyright (C) The Zhejiang University DMAC Lab Authors. team - All Rights Reserved
#
# Written by Xiangji Zeng <xijiz@qq.com>, March 2020
"""Evaluation Metrics"""

from typing import List, Dict, Any
from sklearn.metrics import f1_score, precision_score, recall_score
from .common import to_entities


def evalner(batch_gold_labels: List[List[str]], batch_pred_labels: List[List[str]]) -> Dict[str, Any]:
    """
    Calculate NER metrics: F1, Precision, Recall.

    Args:
        batch_gold_labels (List[List[str]]): Batch gold labels with shape (batch_size, seq_len).
        batch_pred_labels (List[List[str]]): Batch predicted labels with shape (batch_size, seq_len).
    """
    # nsp: not successfully predicted
    scores = {"nsp": []}
    # entity level
    all_gold_entities, all_pred_entities, all_correct_entities = [], [], []
    for i, _ in enumerate(batch_gold_labels):
        gold_entities = [
            "[{0}]({1},{2})".format(span["label"], span["start"], span["end"])
            for span in to_entities("O"*len(batch_gold_labels[i]), batch_gold_labels[i])
        ]
        pred_entities = [
            "[{0}]({1},{2})".format(span["label"], span["start"], span["end"])
            for span in to_entities("O"*len(batch_pred_labels[i]), batch_pred_labels[i])
        ]
        correct_set = set(gold_entities).intersection(set(pred_entities))
        scores["nsp"].append(list(set(gold_entities) - correct_set))
        all_correct_entities.extend(list(correct_set))
        all_gold_entities.extend(gold_entities)
        all_pred_entities.extend(pred_entities)
    scores["entity"] = {
        "p": len(all_correct_entities) / len(all_pred_entities) if len(all_pred_entities) > 0 else 0,
        "r": len(all_correct_entities) / len(all_gold_entities) if len(all_gold_entities) > 0 else 0
    }
    deno = scores["entity"]["p"] + scores["entity"]["r"]
    scores["entity"]["f1"] = 2*scores["entity"]["p"]*scores["entity"]["r"] / deno if deno > 0 else 0

    # token level
    gold_labels, pred_labels = [], []
    for i, _ in enumerate(batch_gold_labels):
        gold_labels.extend(batch_gold_labels[i])
        pred_labels.extend(batch_pred_labels[i])
    scores["token"] = {
        "p": precision_score(gold_labels, pred_labels, average='micro'),
        "r": recall_score(gold_labels, pred_labels, average='micro'),
        "f1": f1_score(gold_labels, pred_labels, average='micro')
    }

    return scores
