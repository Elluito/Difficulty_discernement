from uncertainty_estimation import get_uncertainty_per_image
import config_bayesian as cfg
from main_bayesian import get_sample_distribution
import matplotlib.pyplot as plt
import torch
import numpy as np
import data
from main_bayesian import getModel
import os
import argparse

import torch
import numpy as np
from torch.optim import Adam, lr_scheduler
from torch.nn import functional as F

import data
import utils
import metrics
import tqdm
import config_bayesian as cfg


def main(dataset,net_type):
    # Hyper Parameter settings
    device = torch.device("cuda")
    layer_type = cfg.layer_type
    activation_type = cfg.activation_type
    priors = cfg.priors

    train_ens = cfg.train_ens
    valid_ens = cfg.valid_ens
    n_epochs = cfg.n_epochs
    lr_start = cfg.lr_start
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size
    beta_type = cfg.beta_type

    trainset, testset, inputs, outputs = data.getDataset(dataset)
    train_loader, valid_loader, test_loader = data.getDataloader(
        trainset, testset, valid_size, batch_size, num_workers)
    net = getModel(net_type, inputs, outputs, priors, layer_type, activation_type).to(device)
    ckpt_dir = f'checkpoints/{dataset}/bayesian'
    ckpt_name = f'checkpoints/{dataset}/bayesian/model_{net_type}_{layer_type}_{activation_type}.pt'
    criterion = metrics.ELBO(len(trainset)).to(device)
    # Load the trained net
    net.load_state_dict(torch.load(ckpt_name))
    # Get the good and badly classified samples
    goodly_classified, badly_classified, accuracy = get_sample_distribution(net, criterion=criterion,
                                                                            test_dataset=trainset,
                                                                            num_ens=10, epoch=cfg.n_epochs,
                                                                            num_epochs=cfg.n_epochs)
    ub = max(np.max(goodly_classified), np.max(badly_classified))
    lb = min(np.min(goodly_classified), np.min(badly_classified))
    bins = np.linspace(lb, ub, 100)
    plt.hist(goodly_classified, bins, alpha=0.5, label="Easy examples")
    plt.hist(badly_classified, bins, alpha=0.5, label="Hard examples")
    plt.legend(loc='upper right')
    plt.savefig("experiments/figures/histogram_hard_easy_examples.png")


if __name__ == '__main__':
    main("CIFAR10","3conv3fc")
