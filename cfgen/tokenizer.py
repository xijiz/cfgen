# coding=utf-8
# Copyright (C) The Zhejiang University DMAC Lab Authors. team - All Rights Reserved
#
# Written by Xiangji Zeng <xijiz@qq.com>, March 2020
"""Tokenizer"""

import os
from typing import Dict, Any, List, Union
from collections import OrderedDict
import json
import random
import torch
import shutil
from .common import save_text


class BaseTokenizer:
    """
    BaseTokenizer defines basic interfaces and implements common functions of a Tokenizer.

    Args:
        folder (str): The folder where all token files exist.
        urls (List[str]): A list of pretrained token file links. 
    """

    def __init__(self, folder: str, urls: List[str]):
        super(BaseTokenizer, self).__init__()
        self._folder = folder
        self._urls = urls
        self._to_ids = {}
        self._to_tokens = []
        self._load()
        assert all([sptoken in self._to_ids.keys() for sptoken in ["[UNK]", "[PAD]", "[EMPTY]"]])

    def save(self, folder: str) -> None:
        """
        Save tokens and configuations into a given folder.

        Args:
            folder (str): Saving destination.
        """
        shutil.copyfile(os.path.join(self._folder, "tokens.txt"), os.path.join(folder, "tokens.txt"))
        # shutil.copyfile(os.path.join(self._folder, "embeddings.checkpoints"), os.path.join(folder, "embeddings.checkpoints"))
        shutil.copyfile(os.path.join(self._folder, "configs.json"), os.path.join(folder, "configs.json"))

    def _load(self) -> None:
        """
        Load all token mapping.
        """
        file_names = ["tokens.txt", "embeddings.checkpoints", "configs.json"]
        raw_folder = os.path.join(self._folder, "raw")
        if not os.path.isdir(self._folder) or not os.path.isdir(raw_folder) or len(os.listdir(raw_folder)) == 0:
            if not os.path.isdir(raw_folder):
                os.makedirs(raw_folder)
            print("Tokenization data is not found, please download data into {0}.".format(raw_folder))
            for url in self._urls:
                print(url)
            exit(0)
        if len(set(os.listdir(self._folder)).intersection(set(file_names))) != len(file_names):
            results = self._preprocess()
            assert all([key in file_names for key in results.keys()]) and len(results.keys()) == 3
            for key, value in results.items():
                if key == "tokens.txt" and isinstance(value, list):
                    save_text("\n".join(value), self._folder, "tokens.txt")
                elif key == "configs.json" and isinstance(value, dict):
                    save_text(json.dumps(value, ensure_ascii=False), self._folder, "configs.json")
                elif key == "embeddings.checkpoints" and (isinstance(value, torch.FloatTensor) or isinstance(value, OrderedDict)):
                    torch.save(value, os.path.join(self._folder, "embeddings.checkpoints"))
                else:
                    raise ValueError("incorrect object {0} with type {1}".format(key, type(value)))
        with open(os.path.join(self._folder, "tokens.txt"), "r", encoding="utf-8") as f_in:
            for line in f_in.readlines():
                token = line.replace("\n", "")
                if token != "":
                    self._to_ids[token] = len(self)
                    self._to_tokens.append(token)

    def __len__(self) -> int:
        """Return the total number of tokens."""
        return len(self._to_tokens)

    def __getitem__(self, idx: Union[int, str]) -> Union[str, int]:
        """
        Convert token into ID or ID into token.

        Args:
            idx (Union[int, str]): ID or label.
        """
        if isinstance(idx, int) and 0 <= idx < len(self):
            return self._to_tokens[idx]
        if isinstance(idx, str) and idx in self._to_ids.keys():
            return self._to_ids[idx]

        return self._to_ids[self.unk_token]

    @property
    def pad_token(self) -> str:
        """Return PAD token."""
        return "[PAD]"

    @property
    def pad_id(self) -> int:
        """Return PAD ID."""
        return self[self.pad_token]

    @property
    def unk_token(self) -> str:
        """Return UNK token."""
        return "[UNK]"

    @property
    def unk_id(self) -> int:
        """Return UNK ID."""
        return self[self.unk_token]

    @property
    def empty_token(self) -> str:
        """Return EMPTY token."""
        return "[EMPTY]"

    @property
    def empty_id(self) -> int:
        """Return EMPTY ID."""
        return self[self.empty_token]

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Convert a list of tokens into IDs.

        Args:
            tokens (List[str]): A list of tokens.
        """
        return [self[token] for token in tokens]

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a text into a list of tokens.

        Args:
            text (str): The text to be tokenized.
        """
        return list(text)

    def token_embeddings(self) -> torch.FloatTensor:
        """Return token embeddings."""
        return torch.load(os.path.join(self._folder, "embeddings.checkpoints"))

    def configs(self) -> Dict[str, Any]:
        """Return token configurations."""
        with open(os.path.join(self._folder, "configs.json"), "r", encoding="utf-8") as f_in:
            content = json.loads(f_in.read())
        return content

    def _preprocess(self) -> Dict[str, Any]:
        """Preprocess tokens, embeddings, and configurations."""
        raise NotImplementedError


class TCETokenizer(BaseTokenizer):
    """
    Tokenizer for Tencent AILab Chinese Embedding.
    """

    def __init__(self, folder: str):
        super(TCETokenizer, self).__init__(
            folder,
            [
                "https://ai.tencent.com/ailab/nlp/en/embedding.html",
                "https://ai.tencent.com/ailab/nlp/en/data/Tencent_AILab_ChineseEmbedding.tar.gz"
            ]
        )

    def _preprocess(self):
        """Preprocess tokens, embeddings, and configurations."""
        raw_folder = os.path.join(self._folder, "raw")
        shutil.unpack_archive(os.path.join(raw_folder, "Tencent_AILab_ChineseEmbedding.tar.gz"), raw_folder)
        raw_file_name = os.path.join(raw_folder, "Tencent_AILab_ChineseEmbedding.txt")
        tokens, embeddings, token_dim = [], [], 0

        with open(raw_file_name, "r", encoding="utf-8") as f_in:
            line = f_in.readline()
            token_dim = int(line.split(" ")[1])
            line = f_in.readline()
            while line:
                elems = line.split(" ")
                token, embedding = elems[0], [float(tok) for tok in elems[1:]]
                if len(token) == 1 and len(embedding) == token_dim:
                    tokens.append(token)
                    embeddings.append(embedding)
                line = f_in.readline()
        tokens.extend(["[UNK]", "[EMPTY]", "[PAD]"])
        embeddings.extend([[random.uniform(0.0, 1.0) for _ in range(token_dim)], [0.0]*token_dim, [0.0]*token_dim])
        embeddings = torch.tensor(embeddings)
        configs = {"n_tokens": len(tokens), "token_dim": token_dim}

        return {"tokens.txt": tokens, "configs.json": configs, "embeddings.checkpoints": embeddings}


class BERTTokenizer(BaseTokenizer):
    """
    Tokenizer for BERT base Chinese.
    """

    def __init__(self, folder: str):
        super(BERTTokenizer, self).__init__(
            folder,
            [
                "https://huggingface.co/bert-base-chinese#",
                "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-config.json",
                "https://cdn.huggingface.co/bert-base-chinese-pytorch_model.bin",
                "https://cdn.huggingface.co/bert-base-chinese-vocab.txt"
            ]
        )

    def _preprocess(self):
        """Preprocess tokens, embeddings, and configurations."""
        # we only take two layers of BERT model.
        n_layers = 2
        raw_folder = os.path.join(self._folder, "raw")
        # pretrained weights
        layer_names = tuple(["encoder.layer.{0}.".format(i) for i in range(n_layers)])
        original_embeddings, embeddings = torch.load(os.path.join(raw_folder, "bert-base-chinese-pytorch_model.bin")), OrderedDict()
        for key, value in original_embeddings.items():
            if not key.startswith("cls"):
                key = key.replace("bert.", "")
                if key.startswith("encoder.layer.") and not key.startswith(layer_names):
                    continue
                key = key.replace("embeddings.word_embeddings", "embeddings.token_embeddings")
                key = key.replace("LayerNorm", "layer_norm")
                key = key.replace("gamma", "weight")
                key = key.replace("beta", "bias")
                embeddings[key] = value
        # tokens
        with open(os.path.join(raw_folder, "bert-base-chinese-vocab.txt"), "r", encoding="utf-8") as f_in:
            tokens = [line.replace("\n", "") for line in f_in.readlines()]
        tokens[99] = "[EMPTY]"
        # configs
        with open(os.path.join(raw_folder, "bert-base-chinese-config.json"), "r", encoding="utf-8") as f_in:
            original_configs = json.loads(f_in.read())

        configs = {
            "attention_dropout": original_configs["attention_probs_dropout_prob"],
            "hidden_dropout": original_configs["hidden_dropout_prob"],
            "hidden_dim": original_configs["hidden_size"],
            "intermediate_size": original_configs["intermediate_size"],
            "layer_norm_eps": original_configs["layer_norm_eps"],
            "max_seq_len": original_configs["max_position_embeddings"],
            "n_heads": original_configs["num_attention_heads"],
            "n_layers": n_layers,
            "n_token_types": original_configs["type_vocab_size"],
            "n_tokens": original_configs["vocab_size"],
            "padding_id": tokens.index("[PAD]")
        }

        return {"tokens.txt": tokens, "configs.json": configs, "embeddings.checkpoints": embeddings}
