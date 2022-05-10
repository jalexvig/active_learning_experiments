import numpy as np
from matplotlib import animation, pyplot as plt
import seaborn as sns

from active_learning.utils import get_heatmap_labels, get_heatmap_to_plot


def uncertainty_at_different_withhelds(data, uncertainties, withheld_idxs):

    vmin = np.concatenate(uncertainties).min()
    vmax = np.concatenate(uncertainties).max()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    def animate(tup):
        i, (epistemic, withheld_idxs) = tup

        fig.suptitle(len(withheld_idxs), fontsize=16)

        mask = get_heatmap_labels(data, withheld_idxs)
        masked_to_plot = get_heatmap_to_plot(data, mask)
        g1 = sns.heatmap(
            masked_to_plot, cmap="viridis", vmin=0, vmax=1, ax=ax1, cbar=i == 0
        )
        g1.set_ylabel("")
        g1.set_xlabel("")

        epistemic_to_plot = get_heatmap_to_plot(data, epistemic)
        g2 = sns.heatmap(
            epistemic_to_plot, cmap="viridis", vmin=vmin, vmax=vmax, ax=ax2, cbar=i == 0
        )
        g2.set_ylabel("")
        g2.set_xlabel("")

    tups = enumerate(zip(uncertainties, withheld_idxs))
    anim = animation.FuncAnimation(
        fig, animate, frames=tups, interval=1000, repeat=False
    )
    return anim
