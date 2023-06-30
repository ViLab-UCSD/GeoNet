### img_flist.py
# A dataset class to load image file list from a txt/csv file.
# Author: Gina Wu @ 01/22
###

import os
import torch.utils.data as data

from .utils import flist_reader, default_loader

class ImageFilelist(data.Dataset):

    def __init__(self, root_dir, flist, transform=None, target_transform=None,
                 flist_reader=flist_reader, loader=default_loader):

        self.root_dir = root_dir
        self.data, self.target = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.data[index]
        target = self.target[index]

        impath = os.path.join(self.root_dir, impath)
        img = self.loader(impath)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


