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

        smplr_dict = cfg.get('sampler', {'name': 'random'})

        self.imbalance_factor = cfg.get('imbalance_factor', 1.0)
        self.reverse = cfg.get('reversed', False)
        self.mode = cfg.get('mode', 'exp')

        self.data_loader = dict()
        for split in splits:

            data_list = cfg.get(split, None)
            if not os.path.isfile(data_list):
                raise Exception('{} not available'.format(data_list))

            trans = transform(split)
            shuffle = split == "train"

            dataset = self.getDataset(root_dir=data_root, flist=data_list, transform=trans, loader=default_loader)

            if ('train' in split):
                n_class = cfg.get('n_class', 345)
                cls_num_list = []
                counter = Counter(dataset.target)
                for i in range(n_class):
                    cls_num_list.append(counter.get(i, 1e-7))
                
                if self.mode in ["exp" , "step"]:
                    cls_num_list = self.generate_long_tail_dataset(dataset, cls_num_list)
                elif self.mode in ["prune"]:
                    cls_num_list = self.prune_dataset(dataset, cls_num_list)
                self.data_loader['cls_num_list'] = np.asarray(cls_num_list)

            drop_last = cfg.get('drop_last', False) if 'train' in split else False
            if ('train' in split) and (smplr_dict['name'] != 'random'):
                sampler = get_sampler(dataset, smplr_dict)
                self.data_loader[split] = data.DataLoader(
                    dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
                    drop_last=drop_last, pin_memory=True, num_workers=num_workers
                )
            else:
                self.data_loader[split] = data.DataLoader(
                    dataset, batch_size=batch_size,  sampler=None, shuffle=shuffle,
                    drop_last=drop_last, pin_memory=False, num_workers=num_workers
                )

                # for _,_,idx in self.data_loader[split]:
                #     print(idx.sum())
                #     import pdb; pdb.set_trace()


            # ## inverse weighted sampler.
            # if ('train' in split) and cfg["inverse"]:
            #     sampler = self.get_inverse_weighted_sampler(dataset, cls_num_list)
            #     self.data_loader[split+"_inverse"] = data.DataLoader(
            #         dataset, batch_size=batch_size,  sampler=sampler, shuffle=False,
            #         drop_last=drop_last, pin_memory=True, num_workers=num_workers
            #     )

            logger.info("{split}: {size}".format(split=split, size=len(dataset)))

    def getDataset(self):

        raise NotImplementedError

    def get_inverse_weighted_sampler(self, dataset, cls_num_list, weighted_alpha=1.):
        cls_weight = 1.0 / (np.array(cls_num_list) ** weighted_alpha)
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        samples_weight = np.array([cls_weight[t] for t in dataset.target])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        print("samples_weight", samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(dataset.target), replacement=True)
        return sampler

    def prune_dataset(self, dataset, cls_num_list):
        """
        Prune same fraction of data from all classes.
        """
        prune_frac = self.imbalance_factor
        if prune_frac == 1:
            return cls_num_list

        num_samples = np.ceil(np.array([n*prune_frac for n in cls_num_list]))

        select_idxs = []
        for label_id , n_s in enumerate(num_samples):
            class_indices = np.where(np.array(dataset.target) == label_id)[0]
            np.random.shuffle(class_indices)
            select_idx_label = class_indices[:int(n_s)]
            select_idxs.extend(select_idx_label.tolist())

        ## subsample both data and target
        dataset.data = [dataset.data[idx] for idx in select_idxs]
        dataset.target = [dataset.target[idx] for idx in select_idxs]
        
        new_cls_num_list = []
        n_classes = len(cls_num_list)
        counter = Counter(dataset.target)
        for i in range(n_classes):
            new_cls_num_list.append(counter.get(i, 1e-7))

        return new_cls_num_list

    def generate_long_tail_dataset(self, dataset, cls_num_list):
        """
        Generate long tail version of the dataset with given imbalance factors
        Uses exponential sampling from first class to last class.
        Modifies dataset inplace.
        """

        if self.imbalance_factor == 1:
            return cls_num_list
        
        max_images = max(cls_num_list)

        if self.mode == "exp":
            n_classes = len(cls_num_list)
            if self.reverse:
                exp_factor = np.array([(n_classes-1-i)/(n_classes-1) for i in range(n_classes)])
            else:
                exp_factor = np.array([i/(n_classes-1) for i in range(n_classes)])
            num_samples = np.ceil(max_images * (self.imbalance_factor**exp_factor))
        elif self.mode == "step":
            raise NotImplementedError
        else:
            return 

        select_idxs = []

        for label_id , n_s in enumerate(num_samples):
            class_indices = np.where(np.array(dataset.target) == label_id)[0]
            np.random.shuffle(class_indices)
            select_idx_label = class_indices[:int(n_s)]
            select_idxs.extend(select_idx_label.tolist())

        ## subsample both data and target
        dataset.data = [dataset.data[idx] for idx in select_idxs]
        dataset.target = [dataset.target[idx] for idx in select_idxs]
        
        new_cls_num_list = []
        counter = Counter(dataset.target)
        for i in range(n_classes):
            new_cls_num_list.append(counter.get(i, 1e-7))

        return new_cls_num_list
        

# unit-test
if __name__ == '__main__':
    import pdb
    cfg = {
        'data_root': '/data8/gina/dataset/DomainNet',
        'train': '/data8/gina/dataset/DomainNet/clipart_train.txt',
        'val': '/data8/gina/dataset/DomainNet/clipart_test.txt',
        'n_workers': 4,
        'sampler': {'name': 'class_balanced'}
    }
    splits = ['train', 'val']
    data_loader = BaseLoader(cfg, splits, batch_size=4)
    for (step, value) in enumerate(data_loader['train']):
        img, label = value
        pdb.set_trace()