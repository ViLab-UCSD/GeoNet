from loader.baseloader import BaseLoader
from loader.img_flist import ImageFilelist
from loader.json_loader import ImageJSONLoader
from torchvision.datasets import ImageFolder

import logging

logger = logging.getLogger('mylogger')

class JsonDataLoader(BaseLoader):
    """Function to build data loader(s) for the specified splits given the parameters.
    """

    def __init__(self, cfg, splits, batch_size):
        super().__init__(cfg, splits, batch_size)

    def getDataset(self, root_dir, json_dir, transform=None, loader=None, **kwargs):
        return ImageJSONLoader(root_dir=root_dir, json_path=json_dir, transform=transform, **kwargs)

class FileDataLoader(BaseLoader):
    """Function to build data loader(s) for the specified splits given the parameters.
    """

    def __init__(self, cfg, splits, batch_size):
        super().__init__(cfg, splits, batch_size)

    def getDataset(self, root_dir, flist=None, transform=None, loader=None, **kwargs):
        return ImageFilelist(root_dir=root_dir, flist=flist, transform=transform, **kwargs)

class ImageDataLoader(BaseLoader):
    """Function to build data loader(s) for the specified splits given the parameters.
    """

    def __init__(self, cfg, splits, batch_size):
        super().__init__(cfg, splits, batch_size)

    def getDataset(self, data_root, data_list=None, trans=None, loader=None, **kwargs):
        dataset = ImageFolder(root_dir=data_root, transform=trans, loader=loader, **kwargs)    
        dataset.data = [imgs[0] for imgs in dataset.imgs]
        dataset.target = [imgs[1] for imgs in dataset.imgs]
        dataset.root_dir = dataset.root

        return dataset


# unit-test
if __name__ == '__main__':
    import pdb
    cfg = {
        'data_root': '/newfoundland/tarun/datasets/Adaptation/visDA/',
        'train': 'data/visDA_full/clipart_train.txt',
        'val': 'data/visDA_full/real_train.txt',
        'n_workers': 4,
        'sampler': {'name': 'random'}
    }
    splits = ['train', 'val']
    data_loader = FileDataLoader(cfg, splits, batch_size=4)
    import pdb; pdb.set_trace()
    for (step, value) in enumerate(data_loader['train']):
        img, label = value
        pdb.set_trace()


