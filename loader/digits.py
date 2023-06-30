import os
import logging
from torchvision import transforms
import torch.utils.data as data
import torchvision.datasets as datasets

logger = logging.getLogger('mylogger')

_all_ = ["MNISTLoader" , "SVHNLoader" , "USPSLoader"]

def MNISTLoader(cfg, splits, batch_size):
    """Function to build data loader(s) for MNIST/MNIST-M/SVHN class.
    """

    data_root = cfg.get('data_root', '/path/to/dataset')
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    # if not exist, download mnist dataset
    train_set = datasets.MNIST(root=data_root, train=True, transform=trans, download=True)
    test_set = datasets.MNIST(root=data_root, train=False, transform=trans, download=True)


    if not os.path.isdir(data_root):
        raise Exception('{} does not exist'.format(data_root))

    num_workers = cfg.get('n_workers', 4)

    data_loader = dict()

    data_loader['train'] = data.DataLoader(
                train_set, batch_size=batch_size,  sampler=None, shuffle=True,
                drop_last=False, pin_memory=True, num_workers=num_workers
            )
    logger.info("{split}: {size}".format(split="train", size=len(train_set)))

    data_loader['val'] = data.DataLoader(
                test_set, batch_size=batch_size,  sampler=None, shuffle=False,
                drop_last=False, pin_memory=True, num_workers=num_workers
            )

    logger.info("{split}: {size}".format(split="val", size=len(test_set)))

    logger.info("Building data loader with {} workers".format(num_workers))

    return data_loader

def SVHNLoader(cfg, splits, batch_size):
    """Function to build data loader(s) for MNIST/MNIST-M/SVHN class.
    """

    data_root = cfg.get('data_root', '/path/to/dataset')
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    # if not exist, download mnist dataset
    train_set = datasets.SVHN(root=data_root, split="train", transform=trans, download=True)
    test_set = datasets.SVHN(root=data_root, split="test", transform=trans, download=True)


    if not os.path.isdir(data_root):
        raise Exception('{} does not exist'.format(data_root))

    num_workers = cfg.get('n_workers', 4)

    data_loader = dict()

    data_loader['train'] = data.DataLoader(
                train_set, batch_size=batch_size,  sampler=None, shuffle=True,
                drop_last=False, pin_memory=True, num_workers=num_workers
            )
    logger.info("{split}: {size}".format(split="train", size=len(train_set)))

    data_loader['val'] = data.DataLoader(
                test_set, batch_size=batch_size,  sampler=None, shuffle=False,
                drop_last=False, pin_memory=True, num_workers=num_workers
            )

    logger.info("{split}: {size}".format(split="val", size=len(test_set)))

    logger.info("Building data loader with {} workers".format(num_workers))

    return data_loader

def USPSLoader(cfg, splits, batch_size):
    """Function to build data loader(s) for MNIST/MNIST-M/SVHN class.
    """

    data_root = cfg.get('data_root', '/path/to/dataset')
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    # if not exist, download mnist dataset
    train_set = datasets.USPS(root=data_root, train=True, transform=trans, download=True)
    test_set = datasets.USPS(root=data_root, train=False, transform=trans, download=True)


    if not os.path.isdir(data_root):
        raise Exception('{} does not exist'.format(data_root))

    num_workers = cfg.get('n_workers', 4)

    data_loader = dict()

    data_loader['train'] = data.DataLoader(
                train_set, batch_size=batch_size,  sampler=None, shuffle=True,
                drop_last=False, pin_memory=True, num_workers=num_workers
            )
    logger.info("{split}: {size}".format(split="train", size=len(train_set)))

    data_loader['val'] = data.DataLoader(
                test_set, batch_size=batch_size,  sampler=None, shuffle=False,
                drop_last=False, pin_memory=True, num_workers=num_workers
            )

    logger.info("{split}: {size}".format(split="val", size=len(test_set)))

    logger.info("Building data loader with {} workers".format(num_workers))

    return data_loader