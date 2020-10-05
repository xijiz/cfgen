# coding=utf-8
# Copyright (C) The Zhejiang University DMAC Lab Authors. team - All Rights Reserved
#
# Written by Xiangji Zeng <xijiz@qq.com>, March 2020
"""Counterfactual Generator"""

import copy
import random
from typing import List
from collections import defaultdict


def create_counterfactual_examples(trainset: List[dict]) -> List[dict]:
    """
    Given a part of observational seed examples, creating counterfactual examples.

    Args:
        trainset (List[dict]): A list of observational seed examples.
    """
    deduplicated_examples = set()
    counterfactual_examples = []
    local_entity_sets = defaultdict(set)
    for example in trainset:
        deduplicated_examples.add(copy.deepcopy(example["text"]))
        example["spans"] = sorted(example["spans"], key=lambda s: s["start"], reverse=False)
        for span_id, span in enumerate(example["spans"]):
            local_entity_sets[span["label"]].add(span["text"])

    for i, example in enumerate(trainset):
        if not example["spans"]:
            continue
        index = random.choice(list(range(len(example["spans"]))))
        for local_candidate in local_entity_sets[example["spans"][index]["label"]]:
            cfexample = copy.deepcopy(example)
            cfexample["obersavational_text"] = example["text"]
            cfexample["text"] = "{0}{1}{2}".format(
                cfexample["text"][: cfexample["spans"][index]["start"]],
                local_candidate,
                cfexample["text"][cfexample["spans"][index]["end"] + 1:]
            )
            if cfexample["text"] == example["text"] or cfexample["text"] in deduplicated_examples:
                continue
            deduplicated_examples.add(copy.deepcopy(cfexample["text"]))
            dist = len(local_candidate) - len(cfexample["spans"][index]["text"])
            cfexample["spans"][index]["end"] = cfexample["spans"][index]["start"] - 1 + len(local_candidate)
            cfexample["spans"][index]["text"] = local_candidate
            cfexample["replaced"] = [
                "[{0}]({1}, {2})".format(
                    cfexample["spans"][index]["label"], cfexample["spans"][index]["start"], cfexample["spans"][index]["end"]
                )
            ]
            for i in range(index + 1, len(cfexample["spans"])):
                cfexample["spans"][i]["start"] += dist
                cfexample["spans"][i]["end"] += dist
            counterfactual_examples.append(cfexample)

    return counterfactual_examples
