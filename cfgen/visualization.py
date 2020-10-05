# coding=utf-8
# Copyright (C) The Zhejiang University DMAC Lab Authors. team - All Rights Reserved
#
# Written by Xiangji Zeng <xijiz@qq.com>, March 2020
"""Visualization"""

import os


def visualize_avearage_causal_effects(folder: str, file_name: str = "intervention.csv") -> None:
    """
    Visualize Average Causal Effects.

    References:
        [1] https://seaborn.pydata.org/tutorial/color_palettes.html

    Args:
        folder (str): The folder name where the experimental file exists.
        file_name (str): The experimental file name.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    file_name = os.path.join(folder, file_name)
    data = pd.read_csv(file_name, "\t")
    data = data[data["Level"] == "Token"]
    data.loc[data.Augmentation == 0, "Augmentation"] = "No"
    data.loc[data.Augmentation == 1, "Augmentation"] = "Yes"
    sns.catplot(
        x="N", y="RI", hue="Augmentation", kind="bar", col="Dataset", row="Model",
        palette="Paired", edgecolor="w", linewidth=0.3, aspect = 0.7, data=data
    )
    plt.show()
