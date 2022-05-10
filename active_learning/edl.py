import numpy as np
import torch
from sklearn_extra.cluster import KMedoids
from torch import nn, optim

from deepcure.evaluation.metrics import calculate_regression_metrics
from active_learning.utils import convert_df_to_torch


def NIG_NLL(y, gamma, v, alpha, beta):
    scaling_factor = 2 * beta * (1 + v)

    nll = (
        0.5 * torch.log(np.pi / v)
        - alpha * torch.log(scaling_factor)
        + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + scaling_factor)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    return torch.mean(nll)


def NIG_Reg(y, gamma, v, alpha, beta, omega=0.01, reduce=True):
    error = torch.abs(y - gamma)

    evi = 2 * v + alpha
    evi = 1
    reg = error * evi

    return torch.mean(reg)


def EvidentialRegression(y_true, evidential_output, coeff=1.0):
    gamma, v, alpha, beta = evidential_output.T
    loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta)
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta)
    return loss_nll + coeff * loss_reg


class DenseNormal(nn.Module):
    def __init__(self, size_in):
        super().__init__()
        self.dense = nn.Linear(size_in, 2)

    def forward(self, x):
        output = self.dense(x)
        mu, logsigma = torch.split(output, 1, dim=-1)
        sigma = nn.functional.softplus(logsigma) + 1e-6
        return torch.cat([mu, sigma], dim=-1)


class DenseNormalGamma(nn.Module):
    def __init__(self, size_in):
        super().__init__()
        self.dense = nn.Linear(size_in, 4)

    def forward(self, x):
        output = self.dense(x)
        mu, logv, logalpha, logbeta = torch.split(output, 1, dim=-1)

        v = nn.functional.softplus(logv)
        alpha = nn.functional.softplus(logalpha) + 1
        beta = nn.functional.softplus(logbeta)
        return torch.cat([mu, v, alpha, beta], axis=-1)


def train_model(data, labels, withheld_idxs, num_epochs, seed=0):
    torch.manual_seed(seed)

    train_idxs = data.index.difference(withheld_idxs)
    x_train, y_train = convert_df_to_torch(data, train_idxs), convert_df_to_torch(
        labels, train_idxs
    )

    # Define our model with an evidential output
    model = nn.Sequential(
        nn.Linear(2, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        DenseNormalGamma(64),
    )

    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    def loss_func(true, pred):
        return EvidentialRegression(true, pred, coeff=1)

    #     loss_func = lambda yt, yp: nn.functional.l1_loss(yp[:, 0], yt)

    for epoch in range(num_epochs):

        running_loss = 0

        for x_batch, y_batch in zip(
            torch.split(x_train, 100, dim=0), torch.split(y_train, 100, dim=0)
        ):
            optimizer.zero_grad()
            yp = model(x_batch)
            loss = loss_func(y_batch, yp)
            loss.backward()
            optimizer.step()

            running_loss += loss
    #         if epoch % 10 == 0:
    #             print(running_loss)

    return model


def evaluate_model(model, data, labels, withheld_idxs):
    data = data.loc[withheld_idxs]
    labels = labels.loc[withheld_idxs]
    yp = model(torch.tensor(data.values, dtype=torch.float32))

    res = calculate_regression_metrics(
        labels.values, yp[:, 0].detach().numpy(), high_is_positive=False
    ).iloc[0]

    return res


def choose_new_withheld_dps(model, data, withheld_idxs, strategy, num):
    x = convert_df_to_torch(data, withheld_idxs)
    yp = model(x).detach().numpy()
    mu, v, alpha, beta = yp.T

    if strategy == "random":
        withheld_idxs = np.random.permutation(withheld_idxs)[:-num].tolist()
    elif strategy == "max_uncertainty":
        epistemic = beta / (v * (alpha - 1))
        tups = sorted(zip(epistemic, withheld_idxs))
        withheld_idxs = [idx for _, idx in tups[:-num]]
    elif strategy == "ortho_embeds":
        embeds = model[:-1](x).detach().numpy()
        kmedoids = KMedoids(n_clusters=num, random_state=0, metric="cosine").fit(embeds)
        withheld_idxs = [
            x for i, x in enumerate(withheld_idxs) if i not in kmedoids.medoid_indices_
        ]
    elif strategy == "ortho_uncertainty":
        embeds = model[:-1](x)
        weighted_embeds_per_param = (
            (embeds.unsqueeze(0) * model[-1].dense.weight.unsqueeze(1)).detach().numpy()
        )
        # Concatenate v, alpha, beta since they are used for uncertainty
        weighted_embeds = np.concatenate(weighted_embeds_per_param[1:], axis=-1)
        kmedoids = KMedoids(n_clusters=num, random_state=0, metric="cosine").fit(
            weighted_embeds
        )
        withheld_idxs = [
            x for i, x in enumerate(withheld_idxs) if i not in kmedoids.medoid_indices_
        ]
    else:
        raise ValueError
    return withheld_idxs


def calculate_epistemic(model, data):
    all_x = torch.tensor(data.values, dtype=torch.float32)
    all_yp = model(all_x).detach().numpy()
    mu, v, alpha, beta = all_yp.T

    epistemic = beta / (v * (alpha - 1))
    aleatoric = beta / (alpha - 1)
    return epistemic
