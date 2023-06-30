### __init__.py
# Function to get different types of data loader instances.
# FileDataLoader to load datasets from file - of the form <im_path> <label_id>.
# BaseDataLoader to load datasets from directories of the form images/<class_name>/<im_name>.png.
# 
# Author: Gina Wu @ 01/22
###

from .dataloader import FileDataLoader, ImageDataLoader
from .digits import MNISTLoader, SVHNLoader, USPSLoader


def get_dataloader(cfg, splits, batch_size):
    loader = _get_loader_instance(cfg['loader'])
    data_loader = loader(cfg, splits, batch_size)
    return data_loader.data_loader


def _get_loader_instance(name):
    try:
        return {
            'FileDataLoader': FileDataLoader,
            'ImageDataLoader': ImageDataLoader,
            'MNIST': MNISTLoader,
            'SVHN' : SVHNLoader,
            'USPS' : USPSLoader
        }[name]
    except:
        raise BaseException('Loader type {} not available'.format(name))


