### basedata.py
# Data loader from file
# Author: Gina Wu @ 01/22
###

import os
import logging
import numpy as np
from collections import Counter
import torch

from loader.utils import default_loader
from loader.transforms import transform
from loader.sampler import get_sampler
import torch.utils.data as data


logger = logging.getLogger('mylogger')


class BaseLoader():
    """Function to build data loader(s) for the specified splits given the parameters.
    """

    def __init__(self, cfg, splits, batch_size):
        data_root = cfg.get('data_root', '/path/to/dataset')
        if not os.path.isdir(data_root):
            raise Exception('{} does not exist'.format(data_root))

        num_workers = cfg.get('n_workers', 4)

        self.data_loader = dict()
        for split in splits:

            json_dir = cfg.get("json_dir", None)
            if not os.path.isfile(json_dir):
                raise Exception('{} not available'.format(json_dir))

            trans = transform(split)
            drop_last = cfg.get('drop_last', False) if 'train' in split else False

            kwargs = {
                "domain" : cfg["domain"],
                "return_ann" : cfg.get("ann", "False"),
                "return_loc" : cfg.get("loc", "False"),
                "return_meta" : cfg.get("meta", "False")
            }

            dataset = self.getDataset(root_dir=data_root, json_dir=json_dir, split=split, transform=trans, loader=default_loader, **kwargs)

            if ('train' in split):
                self.data_loader[split] = data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=True,
                    drop_last=drop_last, pin_memory=False, num_workers=num_workers
                )
            else:
                self.data_loader[split] = data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False,
                drop_last=drop_last, pin_memory=False, num_workers=num_workers
            )

            logger.info("{split}: {size}".format(split=split, size=len(dataset)))

    def getDataset(self):

        raise NotImplementedError