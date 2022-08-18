import torch
import logging
import torchvision
import torchmetrics as metrics
import numpy as np
import torchvision as vision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import resnet50
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from sparselearning.utils.accuracy_helper import get_topk_accuracy
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, random_split, Dataset
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import tqdm
import typing
import copy
import random


# efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)

def get_sample_distribution(model: nn.Module, test_dataset: Dataset, device: torch.device):
    g = torch.Generator()
    g.manual_seed(0)
    valLoader = DataLoader(test_dataset, batch_size=128, worker_init_fn=seed_worker,
                           generator=g)

    model.eval()
    model.cuda()
    badly_classified = []
    correctly_classified = []
    counter = 0
    acc = Accuracy()
    pbar = tqdm.tqdm(total=len(valLoader), dynamic_ncols=True)
    with torch.no_grad():
        for inputs, targets in valLoader:
            # assert len(inputs) == 1, "The data-loader does not have batch size 1"
            inputs = inputs.to(device)
            targets = targets.to(device)
            y_pred = model(inputs)
            acc.update((y_pred, targets))
            equal = torch.argmax(y_pred, dim=1) == targets
            for sample_acc in equal:
                if sample_acc > 0:

                    correctly_classified.append(counter)
                    counter += 1

                else:

                    badly_classified.append(counter)

                    counter += 1
            pbar.update(1)

    top_1_accuracy = acc.compute()
    msg = f"Correctly classified: {len(correctly_classified)}, Incorrectly classified: {len(badly_classified)}\n" \
          f"with average accuracy {top_1_accuracy}"

    top_1_accuracy = acc.compute()
    logging.info(msg)
    print(msg)

    return correctly_classified, badly_classified


def fine_tune(model: nn.Module, train_loader, val_loader):
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    from torch.optim.lr_scheduler import ExponentialLR

    lr_scheduler = ExponentialLR(optimizer, gamma=0.975)

    trainer = create_supervised_trainer(model, optimizer, criterion, device="cuda")
    val_metrics = {
        "accuracy": Accuracy(),
        "nll": Loss(criterion)
    }
    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device="cuda")

    @trainer.on(Events.ITERATION_COMPLETED(every=10))
    def log_training_loss(trainer):
        print(f"Epoch[{trainer.state.epoch}] Loss: {trainer.state.output:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print(
            f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print(
            f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")

    print("\nFine tuning has began\n")


    # Setup engine &  logger
    def setup_logger(logger):
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping, TerminateOnNan

    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    # Store the best model
    def default_score_fn(engine):
        score = engine.state.metrics['accuracy']
        return score

    # Force filename to model.pt to ease the rerun of the notebook
    disk_saver = DiskSaver(dirname="models/")
    best_model_handler = Checkpoint(to_save={'efficient-b4': model},
                                    save_handler=disk_saver,
                                    filename_pattern="{name}.{ext}",
                                    n_saved=1)
    evaluator.add_event_handler(Events.COMPLETED, best_model_handler)

    # Add early stopping
    es_patience = 3
    es_handler = EarlyStopping(patience=es_patience, score_function=default_score_fn, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, es_handler)
    setup_logger(es_handler.logger)

    # Clear cuda cache between training/testing
    def empty_cuda_cache(engine):
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: lr_scheduler.step())
    trainer.run(train_loader, max_epochs=100)


def get_vector_of_subset_weights(full_named_parameters: typing.Iterable):
    list_to_convert = []
    names_of_parameters = []
    for name, param in full_named_parameters:
        if "bn" in name:
            continue
        else:
            # Detach and clone are used to avoid any issues with the original model.
            list_to_convert.append(param.data.detach().clone())
            names_of_parameters.append(name)
    vector = parameters_to_vector(list_to_convert)
    dict_with_parameters = dict(zip(names_of_parameters, list_to_convert))
    return vector, dict_with_parameters


def get_weights_of_vector(full_named_parameters: typing.Iterable, vector: torch.Tensor, subset_state_params: dict,
                          ):
    # This is because the "vector" only see the subset_state_dict
    # I need to fill the small_dict with the vector incoming and
    # vector_to_parameters replaces the values inside the iterable he gets.
    # also small_dict has the variable names and the shapes I need for the analysis
    pre_tensors = copy.deepcopy(list(subset_state_params.values()))

    # This is to manage the shapes of each parameter
    vector_to_parameters(vector, pre_tensors)

    # The index is outside because pre_tensors do not have the same amount of elements that full_named_parameters
    index = 0
    for key, param in full_named_parameters:
        if key in subset_state_params.keys():
            # We just want the nen BatchNorm parameters to be copied
            param.data.copy_(pre_tensors[index].clone())
            index += 1


def global_magnitude_prune_vector(vector: torch.Tensor, amount: float = 0.9):
    number_of_zeros = int(len(vector) * amount)
    abs_vector = torch.abs(vector)
    sorting_indexes = torch.argsort(abs_vector)

    prune_vector = torch.ones_like(vector)
    prune_vector[:number_of_zeros] = 0
    vector[sorting_indexes] *= prune_vector


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % -1 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_cifar10():
    BATCH_SIZE = 128
    USABLE_CORES = 1
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    train_transform = transforms.Compose(
        [
            transforms.Pad(4, padding_mode="reflect"),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    # Assign train/val datasets for use in dataloaders
    cifar_full = torchvision.datasets.CIFAR10(".", train=True, download=True,
                                              transform=train_transform)
    cifar10_train, cifar10_val = random_split(cifar_full, [45000, 5000])

    # Assign test dataset mfor use in dataloader(s)
    cifar10_test = torchvision.datasets.CIFAR10(".", train=False, transform=test_transform)

    g = torch.Generator()
    g.manual_seed(0)

    train_loader = DataLoader(cifar10_train, batch_size=BATCH_SIZE, num_workers=USABLE_CORES,
                              worker_init_fn=seed_worker,
                              generator=g, )
    val_loader = DataLoader(cifar10_val, batch_size=BATCH_SIZE, num_workers=USABLE_CORES, worker_init_fn=seed_worker,
                            generator=g)
    test_loader = DataLoader(cifar10_test, batch_size=BATCH_SIZE, num_workers=USABLE_CORES, worker_init_fn=seed_worker,
                             generator=g, )
    return train_loader, val_loader, test_loader, cifar10_test


def get_cifar100():
    BATCH_SIZE = 128
    USABLE_CORES = 1

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    train_transform = transforms.Compose(
        [
            transforms.Pad(4, padding_mode="reflect"),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    # Assign train/val datasets for use in dataloaders
    cifar_full = torchvision.datasets.CIFAR100(".", train=True, download=True,

                                               transform=train_transform)
    cifar10_train, cifar10_val = random_split(cifar_full, [45000, 5000])

    # Assign test dataset mfor use in dataloader(s)
    cifar10_test = torchvision.datasets.CIFAR100(".", train=False, transform=test_transform)

    g = torch.Generator()
    g.manual_seed(0)

    train_loader = DataLoader(cifar10_train, batch_size=BATCH_SIZE, num_workers=USABLE_CORES,
                              worker_init_fn=seed_worker,
                              generator=g, )
    val_loader = DataLoader(cifar10_val, batch_size=BATCH_SIZE, num_workers=USABLE_CORES, worker_init_fn=seed_worker,
                            generator=g)
    test_loader = DataLoader(cifar10_test, batch_size=1, num_workers=USABLE_CORES, worker_init_fn=seed_worker,
                             generator=g, )
    return train_loader, val_loader, test_loader, cifar10_test


def evaluate(model: nn.Module, valLoader: DataLoader, device: torch.device, loss_object: typing.Callable,
             epoch: int, is_test_set: bool = False):
    model.eval()
    model.cuda()
    top1_list = []
    top5_list = []
    loss = 0
    pbar = tqdm.tqdm(total=len(valLoader), dynamic_ncols=True)
    mean = 0
    index = 0
    # valLoader.batch_size = 128
    with torch.no_grad():
        for inputs, targets in valLoader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            y_pred = model(inputs)
            loss += loss_object(y_pred, targets).item()
            top_1_accuracy, top_5_accuracy = get_topk_accuracy(
                y_pred, targets, topk=(1, 5)
            )
            mean = mean + (top_1_accuracy - mean) / (index + 1)
            top1_list.append(top_1_accuracy)
            top5_list.append(top_5_accuracy)
            index += 1
            pbar.set_description(f"Top 1 accuracy: {mean}")
            pbar.update(1)

    loss /= len(valLoader)
    mean_top_1_accuracy = torch.tensor(top1_list).mean()
    mean_top_5_accuracy = torch.tensor(top5_list).mean()

    val_or_test = "val" if not is_test_set else "test"
    msg = f"{val_or_test.capitalize()} Epoch {epoch} {val_or_test} loss {loss:.6f} top-1 accuracy" \
          f" {mean_top_1_accuracy:.4f} top-5 accuracy {mean_top_5_accuracy:.4f}"
    pbar.set_description(msg)
    logging.info(msg)


# def get_params_no_batchnorm(model: nn.Module):
#     lista = []
#     get
#     # modules = list(model.named_modules())
#     # modules.pop(0)
#     # for name, module in modules:
#     #     if isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d) or \
#     #             isinstance(module, torch.nn.BatchNorm2d):
#     #         continue
#     #     else:
#     #
#     #         lista.append((module, "weight"))
#
#     modules = list(model.modules())
#     modules.pop(0)
#     for module in modules:
#         if "bn" in name:
#             continue
#         else:
#
#             lista.append((module, name))
#     return lista
#

def get_collection_of_fast_models(slow_model: nn.Module, pruning="magnitude"):
    if pruning == "magnitude":
        collection = {}
        percentages = torch.linspace(0.1, 0.8, steps=8)
        for per in percentages:
            current_model = copy.deepcopy(slow_model)
            vector_with_no_bn, dict_of_params = get_vector_of_subset_weights(current_model.named_parameters())
            global_magnitude_prune_vector(vector_with_no_bn, amount=per)
            get_weights_of_vector(current_model.named_parameters(), vector_with_no_bn, dict_of_params)
            collection[per] = current_model
        return collection


def get_model(name: str):
    model = None
    if name == "efficientnet":
        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=10)
    if name == "resnet50":
        model = resnet50(pretrained=True, num_classes=10)
    return model


def verify_assumptions(slow_model_samples, fast_models_samples):
    good, bad = slow_model_samples
    set_good_slowModel = set(good)
    good_fast, bad_fast = fast_models_samples
    # This sort begins with the smallest percentage of pruning
    for percent, good_samples in sorted(good_fast.items()):
        set_good_samples = set(good_samples)

        is_smaller = len(set_good_samples) <= len(set_good_slowModel)

        assert is_smaller, f"The number of good examples of the model pruned {percent} times the weights is bigger " \
                           f"than the good examples of the biggest model which violates the assumptions"

        # is_subset = set_good_samples.issubset(set_good_slowModel)
        # is_equal = set_good_samples == set_good_slowModel
        # The symetric difference does not have an element in the small SET.
        condition = len(set_good_samples.symmetric_difference(set_good_slowModel).intersection(set_good_samples))
        assert condition , f"The set of good examples of the model pruned {percent} times the weights is Not a " \
                                      f"subset " \
                                      f"of the set of good examples of the biggest model which violates the assumptions"
    print("All sets of good examples of the pruned networks are subsets of the good examples of the original network")


def main(dataset="", fineTune=True, model_name="efficientnet"):
    if dataset == "cifar10":
        train_loader, val_loader, test_loader, test_dataset = get_cifar10()
        loss_object = F.nll_loss
        model = get_model(model_name)
        # evaluate the model
        # evaluate(model, test_loader, torch.device("cuda"), loss_object, epoch=0, is_test_set=True)
        if fineTune:
            # fine tune the model
            fine_tune(model, train_loader, val_loader)
            # evaluate model again
            evaluate(model, test_loader, torch.device("cuda"), loss_object, epoch=100, is_test_set=True)
            if model_name == "efficientnet":
                torch.save(model.state_dict(), "models/efficientnet-b4_cifar10")
            if model_name == "resnet50":
                torch.save(model.state_dict(), "models/resnet50_cifar10")

        else:

            if model_name == "efficientnet":
                model.load_state_dict(torch.load("models/efficientnet-b4_cifar10"))
        good_original, bad_original = get_sample_distribution(model, test_dataset, torch.device("cuda"))
        collection_of_fast_models = get_collection_of_fast_models(model)
        good_indexes = {}
        bad_indexes = {}
        for percentage, fast_model in collection_of_fast_models.items():
            good, bad = get_sample_distribution(fast_model, test_dataset, torch.device("cuda"))
            good_indexes[percentage] = good
            bad_indexes[percentage] = bad
        verify_assumptions((good_original, bad_original), (good_indexes, bad_indexes))

    if dataset == "cifar100":
        train_loader, val_loader, test_loader, test_dataset = get_cifar100()
        loss_object = f.nll_loss
        model = get_model(model_name)
        # evaluate the model
        evaluate(model, test_loader, torch.device("cuda"), loss_object, epoch=0, is_test_set=True)
        if fineTune:
            # fine tune the model
            fine_tune(model, train_loader, val_loader)
            # evaluate model again
            evaluate(model, test_loader, torch.device("cuda"), loss_object, epoch=100, is_test_set=True)
            if model_name == "efficientnet":
                torch.save(model.state_dict(), "models/efficientnet-b4_cifar100")
            if model_name == "resnet50":
                torch.save(model.state_dict(), "models/resnet50_cifar100")

        else:

            if model_name == "efficientnet":
                model.load_state_dict(torch.load("models/efficientnet-b4_cifar100"))
            if model_name == "resnet50":
                torch.save(model.state_dict(), "models/resnet50_cifar100")


if __name__ == '__main__':
    # test = torch.rand(10)
    # print(f"Test vector pre-pruning: {test}")
    # global_magnitude_prune_vector(test, 0.5)
    # print(f"Test vector pos-pruning: {test}")
    main("cifar10", fineTune=True)
