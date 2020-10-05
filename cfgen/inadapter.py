# coding=utf-8
# Copyright (C) The Zhejiang University DMAC Lab Authors. team - All Rights Reserved
#
# Written by Xiangji Zeng <xijiz@qq.com>, March 2020
"""Input Adapters"""

import os
import random
import json
import shutil
from typing import Dict, Any, List, Tuple

from torch.utils.data import Dataset
from .common import save_text, load_jsonl, split_document
from .tokenizer import BaseTokenizer
from .outadapter import OutAdapter


class BaseInAdapter(Dataset):
    """
    BaseInAdapter provides functions to process various datasets and collate them into a uniform format for feeding
    the neural nets.

    Args:
        data (List[dict]): A list of preprocessed datapoints.
        max_seq_len (int): The maximum length of a sentence.
        outadapter (OutAdapter): Output adapter.
        tokenizer (BaseTokenizer): Tokenizer.
    """

    urls = []

    def __init__(self, data: List[dict], max_seq_len: int, outadapter: OutAdapter, tokenizer: BaseTokenizer):
        super(BaseInAdapter, self).__init__()
        self._dataset = [self.transform_example(example, max_seq_len, outadapter, tokenizer) for example in data]

    @classmethod
    def split_datasets(cls, data_folder: str, max_seq_len: int, is_random: bool = True, testpp: float = 0.1) -> Tuple[list, list, list]:
        """
        Split datasets into three parts: train/dev/test.

        Args:
            data_folder (str): The dataset folder.
            max_seq_len (int): The maximum length of a text.
            is_random (bool): When preprocessing the dataset, whether to ramdom all datapoints.
            testpp (float): The proportion of test set.
        """
        files = set(["data.jsonl", "labels.txt", "unqualified_data.txt", "train.jsonl", "dev.jsonl", "test.jsonl"])
        raw_folder = os.path.join(data_folder, "raw")
        if not os.path.isdir(data_folder) or not os.path.isdir(raw_folder) or len(os.listdir(raw_folder)) == 0:
            if not os.path.isdir(raw_folder):
                os.makedirs(raw_folder)
            print("Data is not found, please download data into {0}.".format(raw_folder))
            for url in cls.urls:
                print(url)
            exit(0)
        if len(set(os.listdir(data_folder)).intersection(files)) != len(files):
            data, labels, unqualified_data = cls._preprocess(data_folder, max_seq_len, is_random)
            n_trains, n_tests = len(data) - int(testpp*2*len(data)), int(testpp*len(data))
            trainset, devset, testset = data[:n_trains], data[n_trains: n_trains + n_tests], data[n_trains + n_tests:]
            save_text("\n".join([json.dumps(sample, ensure_ascii=False) for sample in data]), data_folder, "data.jsonl")
            save_text("\n".join([json.dumps(sample, ensure_ascii=False) for sample in trainset]), data_folder, "train.jsonl")
            save_text("\n".join([json.dumps(sample, ensure_ascii=False) for sample in devset]), data_folder, "dev.jsonl")
            save_text("\n".join([json.dumps(sample, ensure_ascii=False) for sample in testset]), data_folder, "test.jsonl")
            save_text("\n".join(labels), data_folder, "labels.txt")
            save_text("\n".join(unqualified_data), data_folder, "unqualified_data.txt")
            print(
                "{0} data ({1} trains, {2} devs, {3} tests), {4} labels, {5} unqualified_data in {6}"
                .format(len(data), len(trainset), len(devset), len(testset), len(labels), len(unqualified_data), data_folder)
            )
        trainset = load_jsonl(data_folder, "train.jsonl")
        devset = load_jsonl(data_folder, "dev.jsonl")
        testset = load_jsonl(data_folder, "test.jsonl")

        return (trainset, devset, testset)

    @staticmethod
    def _preprocess(data_folder: str, max_seq_len: int, is_random: bool) -> (List[dict], List[str], List[str]):
        """
        Preprocess datasets.

        Args:
            data_folder (str): The dataset folder.
            max_seq_len (int): The maximum length of a text.
            is_random (bool): When preprocessing the dataset, whether to ramdom all datapoints.
        """
        raise NotImplementedError

    def transform_example(self, example: Dict[str, Any], max_seq_len: int, outadapter: OutAdapter, tokenizer: BaseTokenizer) -> Dict[str, Any]:
        """
        Transform the raw datapoint into numbers.

        Args:
            example (Dict[str, Any]): The raw datapoint.
            max_seq_len (int): The maximum length of a sentence.
            outadapter (OutAdapter): The output adapter.
            tokenizer (BaseTokenizer): Tokenizer.
        """
        input_tokens = tokenizer.tokenize(example["text"])[:max_seq_len]
        output_labels = [outadapter.pad_label] * len(input_tokens)
        for span in example["spans"]:
            label = span["label"]
            if span["start"] < len(input_tokens):
                output_labels[span["start"]] = "B-{0}".format(label)
            pointer = span["start"] + 1
            while pointer <= span["end"] and pointer < len(input_tokens):
                output_labels[pointer] = "I-{0}".format(label)
                pointer += 1
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        output_ids = [outadapter[label] for label in output_labels]

        return {"input_ids": input_ids, "output_ids": output_ids, "length": len(input_ids)}

    def __len__(self) -> int:
        """Return the total number of datapoints."""
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return the converted datapoint.

        Args:
            idx (int): The index of the datapoint.
        """
        return self._dataset[idx]


class InCNER(BaseInAdapter):
    """
    Input Adapter for the dataset CNER.
    """

    urls = ["http://www.ccks2019.cn/?page_id=62"]

    @staticmethod
    def _preprocess(data_folder: str, max_seq_len: int, is_random: bool) -> (List[dict], List[str], List[str]):
        """
        Preprocess datasets.

        Args:
            data_folder (str): The dataset folder.
            max_seq_len (int): The maximum length of a text.
            is_random (bool): When preprocessing the dataset, whether to ramdom all datapoints.
        """
        extracted_types = ["疾病和诊断"]
        raw_file_names = ["subtask1_training_part1.txt", "subtask1_training_part2.txt", "subtask1_test_set_with_answer.json"]
        raw_samples = []
        for file_name in raw_file_names:
            file_name = os.path.join(data_folder, "raw", file_name)
            if not os.path.isfile(file_name):
                continue
            with open(file_name, "r", encoding="utf-8-sig") as f_in:
                for line in f_in.readlines():
                    sample = json.loads(line)
                    raw_samples.append((file_name, sample))
        data, unqualified_data = [], []
        labels = ["O"] + ["{}-{}".format(prefix, label) for prefix in ["B", "I"] for label in extracted_types]

        for _, (file_name, sample) in enumerate(raw_samples):
            text, spans, is_normalized = sample["originalText"], [], True
            for span in sample["entities"]:
                if span["label_type"] not in extracted_types:
                    continue
                if "overlap" in span and span["overlap"] > 0:
                    is_normalized = False
                    unqualified_data.append("{0}\t{1}".format(file_name, "overlap in {0}".format(text)))
                    break
                start, end = span["start_pos"], span["end_pos"]
                spans.append({
                    "text": text[start: end],
                    "label": span["label_type"],
                    "start": start,
                    "end": end - 1
                })

            if is_normalized:
                spans.sort(key=lambda ann: ann["start"])
                if len(text) > max_seq_len:
                    samples, ignored = split_document(text, spans, max_seq_len)
                    for sample in ignored:
                        unqualified_data.append("{0}\t{1}".format(file_name, "document too long in {0}".format(sample["text"])))
                else:
                    samples = [{"text": text, "spans": spans}]
                data.extend(samples)

        if is_random:
            random.shuffle(data)

        return (data, labels, unqualified_data)


class InCLUNER(BaseInAdapter):
    """
    Input Adapter for the dataset CLUENER.
    """

    urls = ["https://github.com/chineseGLUE/chineseGLUE"]

    @staticmethod
    def _preprocess(data_folder: str, max_seq_len: int, is_random: bool) -> (List[dict], List[str], List[str]):
        """
        Preprocess datasets.

        Args:
            data_folder (str): The dataset folder.
            max_seq_len (int): The maximum length of a text.
            is_random (bool): When preprocessing the dataset, whether to ramdom all datapoints.
        """
        raw_folder = os.path.join(data_folder, "raw")
        shutil.unpack_archive(os.path.join(raw_folder, "cluener_public.zip"), raw_folder)
        extracted_types = ["address", "book", "company", "game", "government", "movie", "name", "organization", "position", "scene"]
        raw_file_names = ["train.json", "dev.json"]
        data, unqualified_data = [], []
        labels = ["O"] + ["{}-{}".format(prefix, label) for prefix in ["B", "I"] for label in extracted_types]

        for file_name in raw_file_names:
            file_name = os.path.join(data_folder, "raw", file_name)
            with open(file_name, "r", encoding="utf-8") as f_in:
                for line in f_in.readlines():
                    sample = json.loads(line)
                    if len(sample["text"]) > max_seq_len:
                        unqualified_data.append("{0}\texceed maximum length: {1}".format(file_name, sample["text"]))
                        continue
                    spans = []
                    for label, span in sample["label"].items():
                        if label not in extracted_types:
                            unqualified_data.append("{0}\tcan not extract {1} from {2}".format(file_name, label, sample["text"]))
                            continue
                        for text, text_range in span.items():
                            spans.append({
                                "text": text,
                                "label": label,
                                "start": text_range[0][0],
                                "end": text_range[0][1],
                            })
                    data.append({
                        "text": sample["text"],
                        "spans": spans
                    })

        if is_random:
            random.shuffle(data)

        return (data, labels, unqualified_data)
