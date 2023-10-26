from .dataloader import FileDataLoader, ImageDataLoader, JsonDataLoader
from .json_loader import ImageJSONLoader


def get_dataloader(cfg, splits, batch_size):
    loader = _get_loader_instance(cfg['loader'])
    data_loader = loader(cfg, splits, batch_size)
    return data_loader.data_loader


def _get_loader_instance(name):
    try:
        return {
            'FileDataLoader': FileDataLoader,
            'ImageDataLoader': ImageDataLoader,
            'JSONDataLoader': JsonDataLoader,
        }[name]
    except:
        raise BaseException('Loader type {} not available'.format(name))


