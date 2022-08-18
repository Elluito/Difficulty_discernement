############### Configuration file for Bayesian ###############
layer_type = 'lrt'  # 'bbb' or 'lrt'
activation_type = 'softplus'  # 'softplus' or 'relu'
priors={
    'prior_mu': 0,
    'prior_sigma': 0.1,
    'posterior_mu_initial': (0, 0.1),  # (mean, std) normal_
    'posterior_rho_initial': (-5, 0.1),  # (mean, std) normal_
}

n_epochs = 5
lr_start = 0.01
num_workers = 2
valid_size = 0.2
batch_size = 256
train_ens = 3
valid_ens = 3
beta_type = 0.1  # 'Blundell', 'Standard', etc. Use float for const value
