import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn, optim

logger = logging.getLogger()


LIKELIHOOD_EPSILON = 1e-8


@dataclass
class GaussianMixtureModel:

    def __init__(self, num_dims: int, num_mixes: int, learning_rate: float):
        super().__init__()

        self.num_mixes = num_mixes
        self.learning_rate = learning_rate

        self.cgmm = ConditionalGaussianMixtureModule(num_dims, self.num_mixes)
        self.optimizer = optim.Adam(self.cgmm.parameters(), lr=self.learning_rate, weight_decay=0.0)

    def train(self, X, y):

        distrs = self.cgmm(X)

        reshaped = torch.transpose(distrs, 0, 2)
        us, vars_, weights = reshaped

        # Code below will try to force a single mode to be chosen
        # nlls = (torch.log(vars_) + torch.square((y - us) / vars_)) / 2
        #
        # # offset vars_ by 1 so don't go crazy with nll
        # nlls = (torch.log(vars_ + 1) + torch.square((y - us) / (vars_ + 1))) / 2
        #
        # weighted_nll = torch.mean(torch.sum(nlls * weights, axis=0) / torch.sum(weights, axis=0))
        # weighted_nll = torch.mean(nlls)
        # weighted_nll = torch.mean(torch.min(nlls, axis=0)[0])
        # min_idxs = torch.argmin(nlls, axis=0)
        # selected_mode_weights = torch.sum(weights, axis=0) / weights.gather(0, min_idxs[None, :])[0]
        # selected_nlls = nlls.gather(0, min_idxs[None, :])[0]
        # weighted_nll = torch.dot(selected_mode_weights, selected_nlls) / len(selected_nlls)

        likelihood = sum(
            w * torch.exp(-torch.square(y - u) / var / 2) / torch.sqrt(2 * np.pi * var)
            for w, u, var in zip(weights, us, vars_)
        )
        nll = -torch.log(likelihood + LIKELIHOOD_EPSILON).mean()

        self.optimizer.zero_grad()
        # weighted_nll.backward()
        nll.backward()
        self.optimizer.step()

        logger.info('NLL %f', nll)
        return nll

    def evaluate(self, X):

        with torch.no_grad():
            return self.cgmm(X)


class ConditionalGaussianMixtureModule(nn.Module):

    def __init__(self, num_dims: int, num_mixes: int):
        super().__init__()

        self.num_dims = num_dims
        self.num_mixes = num_mixes
        h_sizes = [50, 30]

        self.output_layers = nn.Sequential(
            # TODO(jalex): this is messing up when all 0s. Create custom class to deal with this
            # nn.BatchNorm1d(self.n_distr_dims + self.num_dims),
            nn.Linear(self.n_distr_dims + self.num_dims, h_sizes[0]),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(h_sizes[0]),
            nn.Linear(h_sizes[0], h_sizes[1]),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(h_sizes[1]),
            nn.Linear(h_sizes[1], 3),  # u, var, weight
        )

    @property
    def n_distr_dims(self):
        return 3 * self.num_mixes

    def forward(self, x):
        distrs = torch.zeros(len(x), self.n_distr_dims)

        for i in range(self.num_mixes):
            inp = torch.cat([distrs, x], axis=1)
            distr = self.output_layers(inp)
            distrs[:, i * 3] = distr[:, 0]
            # TODO(jalex): Figure out best way of getting non neg val here for var
            distrs[:, i * 3 + 1] = torch.sqrt(torch.exp(distr[:, 1]))
            distrs[:, i * 3 + 2] = torch.sigmoid(distr[:, 2])

        reshaped = torch.reshape(distrs, (-1, self.num_mixes, 3))

        us, vars_, weights = reshaped.T
        weights = weights / weights.sum(axis=0)

        normalized = torch.stack([us, vars_, weights]).T

        return normalized


