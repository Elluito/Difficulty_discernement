from __future__ import print_function

import os
import argparse

import torch
import numpy as np
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.nn import functional as F

import data
import utils
import metrics
import tqdm
import config_bayesian as cfg
from pre_built__models.BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from pre_built__models.BayesianModels.BayesianAlexNet import BBBAlexNet
from pre_built__models.BayesianModels.BayesianLeNet import BBBLeNet

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % -1 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_uncertainty_per_image(model, input_image, T=15, normalized=False):
    input_image = input_image.unsqueeze(0)
    input_images = input_image.repeat(T, 1, 1, 1)

    net_out, _ = model(input_images)
    pred = torch.mean(net_out, dim=0).cpu().detach().numpy()
    if normalized:
        prediction = F.softplus(net_out)
        p_hat = prediction / torch.sum(prediction, dim=1).unsqueeze(1)
    else:
        p_hat = F.softmax(net_out, dim=1)
    p_hat = p_hat.detach().cpu().numpy()
    p_bar = np.mean(p_hat, axis=0)

    temp = p_hat - np.expand_dims(p_bar, 0)
    epistemic = np.dot(temp.T, temp) / T
    epistemic = np.diag(epistemic)

    aleatoric = np.diag(p_bar) - (np.dot(p_hat.T, p_hat) / T)
    aleatoric = np.diag(aleatoric)

    return pred, epistemic, aleatoric
def getModel(net_type, inputs, outputs, priors, layer_type, activation_type):
    if (net_type == 'lenet'):
        return BBBLeNet(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == 'alexnet'):
        return BBBAlexNet(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == '3conv3fc'):
        return BBB3Conv3FC(outputs, inputs, priors, layer_type, activation_type)
    else:
        raise ValueError('Network should be either [LeNet / AlexNet / 3Conv3FC')


def train_model(net, optimizer, criterion, trainloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    net.train()
    training_loss = 0.0
    accs = []
    kl_list = []
    for i, (inputs, labels) in enumerate(trainloader, 1):

        optimizer.zero_grad()

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)

        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1)

        kl = kl / num_ens
        kl_list.append(kl.item())
        log_outputs = utils.logmeanexp(outputs, dim=2)

        beta = metrics.get_beta(i - 1, len(trainloader), beta_type, epoch, num_epochs)
        loss = criterion(log_outputs, labels, kl, beta)
        loss.backward()
        optimizer.step()

        accs.append(metrics.acc(log_outputs.data, labels))
        training_loss += loss.cpu().data.numpy()
    return training_loss / len(trainloader), np.mean(accs), np.mean(kl_list)


def validate_model(net, criterion, validloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    """Calculate ensemble accuracy and NLL Loss"""
    net.train()
    valid_loss = 0.0
    accs = []

    for i, (inputs, labels) in enumerate(validloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)
        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1).data

        log_outputs = utils.logmeanexp(outputs, dim=2)

        beta = metrics.get_beta(i - 1, len(validloader), beta_type, epoch, num_epochs)
        valid_loss += criterion(log_outputs, labels, kl, beta).item()
        accs.append(metrics.acc(log_outputs, labels))

    return valid_loss / len(validloader), np.mean(accs)


def get_sample_distribution(net, criterion, test_dataset, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    g = torch.Generator()
    g.manual_seed(0)

    validloader = DataLoader(test_dataset, batch_size=cfg.batch_size, worker_init_fn=seed_worker,
                             generator=g)

    net.eval()
    net.to(device)
    valid_loss = 0.0
    accs = []
    badly_classified = {}
    correctly_classified = {}
    counter = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(validloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)
            kl = 0.0
            for j in range(num_ens):
                net_out, _kl = net(inputs)
                kl += _kl
                outputs[:, :, j] = F.log_softmax(net_out, dim=1).data

            log_outputs = utils.logmeanexp(outputs, dim=2)

            beta = metrics.get_beta(i - 1, len(validloader), beta_type, epoch, num_epochs)
            valid_loss += criterion(log_outputs, labels, kl, beta).item()
            batch_accuracy = metrics.acc(log_outputs, labels)
            print(f"batch accuracy: {batch_accuracy}")
            accuracies = outputs.cpu().numpy().argmax(axis=1) == labels.data.cpu().numpy()
            batch_index = 0
            for sample_acc in accuracies:
                if sample_acc > 0:
                    prediction,epistemic,aleatoric = get_uncertainty_per_image(net, inputs[batch_index])
                    correctly_classified[counter] = epistemic
                    counter += 1
                    batch_index += 1

                else:
                    prediction,epistemic,aleatoric = get_uncertainty_per_image(net, inputs[batch_index])
                    badly_classified[counter] = epistemic
                    counter += 1
                    batch_index += 1

            accs.append(metrics.acc(log_outputs, labels))
    return correctly_classified, badly_classified, np.mean(accs)


def run(dataset, net_type):
    # Hyper Parameter settings
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

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    criterion = metrics.ELBO(len(trainset)).to(device)
    optimizer = Adam(net.parameters(), lr=lr_start)
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
    valid_loss_max = np.Inf
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        train_loss, train_acc, train_kl = train_model(net, optimizer, criterion, train_loader, num_ens=train_ens,
                                                      beta_type=beta_type, epoch=epoch, num_epochs=n_epochs)
        valid_loss, valid_acc = validate_model(net, criterion, valid_loader, num_ens=valid_ens, beta_type=beta_type,
                                               epoch=epoch, num_epochs=n_epochs)
        lr_sched.step(valid_loss)

        print(
            'Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f} \ttrain_kl_div: {:.4f}'.format(
                epoch, train_loss, train_acc, valid_loss, valid_acc, train_kl))

        # save model if validation accuracy has increased
        if valid_loss <= valid_loss_max:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_max, valid_loss))
            torch.save(net.state_dict(), ckpt_name)
            valid_loss_max = valid_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Bayesian Model Training")
    parser.add_argument('--net_type', default='lenet', type=str, help='model')
    parser.add_argument('--dataset', default='MNIST', type=str, help='dataset = [MNIST/CIFAR10/CIFAR100]')
    args = parser.parse_args()

    run(args.dataset, args.net_type)
