import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt


def convert_df_to_torch(data, idxs=None, dtype=torch.float32):
    if idxs is not None:
        data = data.loc[idxs]
    return torch.tensor(data.values, dtype=dtype)


def get_heatmap_to_plot(data, z):
    z = pd.Series(z, name='z')
    data = data.round(2)
    dft = pd.concat([data, z], axis=1)
    to_plot = dft.pivot('y', 'x')['z']
    return to_plot


def heatmap_surface(data, z, title=None):
    to_plot = get_heatmap_to_plot(data, z)
    g = sns.heatmap(to_plot, cmap='viridis')
    if title is not None:
        plt.title(title)
    return g


def get_heatmap_labels(data, withheld_idxs):
    mask = pd.Series(0, index=data.index)
    mask.loc[withheld_idxs] = 1
    return mask


def heatmap_withheld(data, withheld_idxs):
    mask = get_heatmap_labels(data, withheld_idxs)
    to_plot = get_heatmap_to_plot(data, mask)
    return heatmap_surface(data, mask)
