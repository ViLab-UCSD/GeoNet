### __init__.py
# Function to get different types of adaptation trainings.
# Author: Tarun Kalluri @ 07/22
###

from .cdan import train_cdan
from .plain import train_plain
from .dann import train_dann
from .val import val
from .eval import eval

def get_trainer(cfg):
    trainer = _get_trainer_instance(cfg['trainer'])
    return trainer


def _get_trainer_instance(name):
    try:
        return {
            'plain' : train_plain,
            'cdan' : train_cdan,
            'dann' : train_dann,
        }[name]
    except:
        raise BaseException('Trainer type {} not available'.format(name))


