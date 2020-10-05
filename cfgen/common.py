# coding=utf-8
# Copyright (C) The Zhejiang University DMAC Lab Authors. team - All Rights Reserved
#
# Written by Xiangji Zeng <xijiz@qq.com>, March 2020
"""Common Utils"""

import json
import os
from typing import Union, Tuple, List, Callable
import torch


def save_text(text: Union[str, dict], *paths: Tuple) -> None:
    """
    Save text into local disk.

    Args:
        text (Union[str, dict]): The text to be saved.
        paths (Tuple): All sub-paths which will be concatenated a complete path.
    """
    if isinstance(text, dict):
        text = json.dumps(text, ensure_ascii="utf-8")
    if not isinstance(text, str):
        raise ValueError("can not convert text into a string")
    file_path = os.path.join(*paths)

    with open(file_path, "w", encoding="utf-8") as f_o:
        f_o.write(text)


def load_jsonl(*file_path: Tuple[str], encoding: str = "utf-8") -> list:
    """
    Load a list of JSON from local disk.

    Args:
        file_path (Tuple): All sub-paths which will be concatenated a complete path.
        encoding (str): Text encoding.
    """
    data = []
    file_path = os.path.join(*file_path)
    with open(file_path, "r", encoding=encoding) as f_in:
        for line in f_in.readlines():
            data.append(json.loads(line))

    return data


def save_jsonl(data: List[dict], *file_path: Tuple[str]) -> None:
    """
    Save a list of JSON into local disk.

    Args:
        data (List[dict]): A list of JSON object.
        file_path (Tuple): All sub-paths which will be concatenated a complete path.
    """
    file_path = os.path.join(*file_path)
    data = [json.dumps(sample, ensure_ascii=False) for sample in data]
    with open(file_path, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(data))


def load_json(*file_path: Tuple[str], encoding: str = "utf-8") -> dict:
    """
    Load a JSON from local disk.

    Args:
        file_path (Tuple): All sub-paths which will be concatenated a complete path.
        encoding (str): Text encoding.
    """
    file_path = os.path.join(*file_path)
    with open(file_path, "r", encoding=encoding) as f_in:
        return json.loads(f_in.read())


def to_entities(text: str, labels: List[str]) -> List[dict]:
    """
    Convert a list of token labels into entities.

    Args:
        text (str): Sentences.
        labels (List[str]): A list of labels.
    """
    spans = []
    i, seq_len = 0, len(labels)
    while i < seq_len:
        if labels[i].startswith("B"):
            label_infos = labels[i].split("-")
            span = {
                "start": i,
                "end": i,
                "label": label_infos[1]
            }
            pointer = i + 1
            if pointer == seq_len:
                span["end"] = pointer - 1
                i = pointer
            while pointer < seq_len:
                if labels[pointer].startswith("O") or labels[pointer].startswith("B"):
                    span["end"] = pointer - 1
                    i = pointer
                    break
                elif pointer == seq_len - 1:
                    span["end"] = pointer
                    i = pointer + 1
                    break
                else:
                    pointer += 1
            span["text"] = text[span["start"]: span["end"] + 1]
            spans.append(span)
        else:
            i += 1

    return spans


def to_tags(seq_len: int, spans: List[dict]) -> list:
    """
    Convert spans to a list of tag for sequence labeling.
        Tag scheme: BIO.

    Args:
        seq_len (int): The length of sequence.
        spans (List[dict]): All spans on this sequence.
    """
    tags = ["O"] * seq_len
    for span in spans:
        pos = span["start"]
        if pos < seq_len:
            tags[pos] = "B-{0}".format(span["label"])
            pos += 1
        while pos < min(span["end"] + 1, seq_len):
            tags[pos] = "I-{0}".format(span["label"])
            pos += 1

    return tags


def split_document(document: str, spans: List[dict], max_seq_len: int = 512, segsym: str = "。") -> Tuple[List[dict], List[dict]]:
    """
    Split long document into sentences with parameter `max_seq_len` as sliding window.

    Args:
        document (str): The raw document.
        spans (List[dict]): Spans on this document. We do not split document inside a span.
        max_seq_len (str): The maximum sequence length of splited sentences.
        segsym (str): The segmentation symbols.
    """
    if document == "":
        return ([], [])
    # this step should be the first, because the following step would change the document length.
    tags = to_tags(len(document), spans)
    if document.endswith(segsym):
        document = document[:len(document)-1]
        sentences = [sentence + segsym for sentence in document.split(segsym)]
    else:
        sentences = [sentence + segsym for sentence in document.split(segsym)]
        sentences[-1] = sentences[-1][:len(sentences[-1])-1]  # remove last segsym
    data, ignored = [], []
    start = 0
    short_document = ""
    for sentence in sentences:
        if len(short_document) + len(sentence) <= max_seq_len:
            short_document += sentence
        else:
            if len(short_document) > 0:
                data.append({
                    "text": short_document,
                    "spans": to_entities(short_document, tags[start: start + len(short_document)])
                })
                start += len(short_document)
                short_document = ""
            if len(sentence) <= max_seq_len:
                short_document = sentence
            else:
                # ignored long fragment
                ignored.append({
                    "text": sentence,
                    "spans": to_entities(sentence, tags[start: start + len(sentence)])
                })
                start += len(sentence)
    if short_document != "":
        if len(short_document) <= max_seq_len:
            data.append({
                "text": short_document,
                "spans": to_entities(short_document, tags[start: start + len(short_document)])
            })
        else:
            ignored.append({
                "text": sentence,
                "spans": to_entities(sentence, tags[start: start + len(sentence)])
            })

    return (data, ignored)


def collate_fn(input_pad_id: int, output_pad_id: int, device: str) -> Callable:
    """
    Collate function for padding batch sentences.

    Args:
        input_pad_id (int): The ID of input pad.
        output_pad_id (int): The ID of output pad.
        device (str): Computing device.
    """
    def collate_fn_wrapper(batch):
        max_seq_len = 0
        for i, _ in enumerate(batch):
            max_seq_len = max(max_seq_len, len(batch[i]["input_ids"]))
        for i, _ in enumerate(batch):
            batch[i]["input_ids"] += [input_pad_id] * (max_seq_len - len(batch[i]["input_ids"]))
            batch[i]["output_ids"] += [output_pad_id] * (max_seq_len - len(batch[i]["output_ids"]))
            batch[i]["masks"] = [1] * batch[i]["length"] + [0] * (max_seq_len - batch[i]["length"])
        input_ids = torch.tensor([sample["input_ids"] for sample in batch], device=device)
        output_ids = torch.tensor([sample["output_ids"] for sample in batch], device=device)
        masks = torch.tensor([sample["masks"] for sample in batch], device=device)

        return (input_ids, output_ids, masks)

    return collate_fn_wrapper
