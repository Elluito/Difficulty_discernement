import torch

from PyTorch_BayesianCNN.uncertainty_estimation import get_uncertainty_per_image
import config_bayesian as cfg
from PyTorch_BayesianCNN.pre_built__models.BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from PyTorch_BayesianCNN.pre_built__models.BayesianModels.BayesianAlexNet import BBBAlexNet
from main import get_cifar10,seed_worker
from PyTorch_BayesianCNN.main_bayesian import get_sample_distribution
import matplotlib.pyplot as plt
def load_model():
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
    criterion = metrics.ELBO(len(trainset)).to(device)
    optimizer = Adam(net.parameters(), lr=lr_start)
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
    valid_loss_max = np.Inf
    # Load the trained net
    net.load_state_dict(torch.load(ckpt_name))
    # Get the good and badly classified samples
    goodly_classified,badly_classified, accuracy = get_sample_distribution(net,criterion=criterion,test_dataset=trainset,
                                                                 num_ens=10,epoch=cfg.n_epochs,num_epochs=cfg.n_epochs)



if __name__ == '__main__':

