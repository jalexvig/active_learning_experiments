import logging
from dataclasses import dataclass

import torch
from torch import nn, optim

logger = logging.getLogger()


@dataclass
class GaussianMixtureModel:

    num_dims: int
    num_mixes: int
    learning_rate: float = 0.001

    def __post_init__(self):
        self.cgmm = ConditionalGaussianMixtureModule(self.num_dims, self.num_mixes)
        self.optimizer = optim.Adam(self.cgmm.parameters(), lr=self.learning_rate, weight_decay=0.0)

    def train(self, X, y):

        distrs = self.cgmm(X)
        # TODO(jalex): The line below is incorrect
        reshaped = torch.reshape(distrs, (-1, 3, self.num_mixes))
        reshaped = torch.transpose(torch.transpose(reshaped, 0, 1), 1, 2)
        us, stds, weights = reshaped
        nlls = torch.log(stds) + torch.square((y - us) / stds) / 2
        weighted_nll = torch.sum(nlls * weights) / torch.sum(weights)

        self.optimizer.zero_grad()
        weighted_nll.backward()
        self.optimizer.step()

        logger.info('Weighted nll %f', weighted_nll)


@dataclass
class ConditionalGaussianMixtureModule(nn.Module):

    num_dims: int
    num_mixes: int

    @property
    def n_distr_dims(self):
        return 3 * self.num_mixes

    def __post_init__(self):

        self.output_layers = nn.ModuleList([
            nn.BatchNorm1d(self.n_distr_dims + self.num_dims),
            nn.Linear(self.num_dims, 50),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(50),
            nn.Linear(50, 3),  # u, std, weight
        ])

    def forward(self, x):
        distrs = torch.zeros(len(x), self.n_distr_dims)

        for i in range(self.num_mixes):
            inp = torch.cat([distrs, x], axis=1)
            distr = self.output_layers(inp)
            # TODO(jalex): Figure out best way of getting non neg val here
            distr[1] = torch.abs(distr[1])
            distr[2] = torch.sigmoid(distr[2])
            distrs[i * 3: (i + 1) * 3] = distr

        return distrs


