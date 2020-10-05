# coding=utf-8
# Copyright (C) The Zhejiang University DMAC Lab Authors. team - All Rights Reserved
#
# Written by Xiangji Zeng <xijiz@qq.com>, March 2020
"""Output Adapter"""

import os
from typing import Union
import shutil


class OutAdapter:
    """
    OutAdapter provides functions to postprocess the data from the neural nets.

    Args:
        folder (str): The data folder where the mapping data exists.
    """

    def __init__(self, folder: str):
        super(OutAdapter, self).__init__()
        self._folder = folder
        self._to_index = {}
        self._to_label = []
        self._load()

    @property
    def pad_id(self) -> int:
        """Return PAD ID."""
        return self[self.pad_label]

    @property
    def pad_label(self) -> str:
        """Return PAD label."""
        return "O"

    def _load(self) -> None:
        """Load all mapping labels."""
        with open(os.path.join(self._folder, "labels.txt"), "r", encoding="utf-8") as f_in:
            data = [line.replace("\n", "") for line in f_in.readlines()]
        for item in data:
            self._to_index[item] = len(self._to_label)
            self._to_label.append(item)

    def __len__(self) -> int:
        """Return the total number of mapping labels."""
        return len(self._to_label)

    def __getitem__(self, item: Union[int, str]) -> Union[str, int]:
        """
        Convert label into ID or ID into label.

        Args:
            item (Union[int, str]): ID or label.
        """
        if isinstance(item, str) and item in self._to_index:
            return self._to_index[item]
        elif isinstance(item, int) and 0 <= item < len(self._to_label):
            return self._to_label[item]
        raise ValueError("No corresponding value found: {0} with type {1}".format(item, type(item)))

    def save(self, folder: str):
        """
        Save labels into a given folder.

        Args:
            folder (str): A given folder to be used for saving labels.
        """
        label_name = "labels.txt"
        shutil.copy(os.path.join(self._folder, label_name), os.path.join(folder, label_name))
