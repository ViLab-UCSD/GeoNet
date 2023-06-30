### utils.py
# Utility functions for data loading.
# Author: Gina Wu @ 01/22
###

import pandas as pd
from PIL import Image

def flist_reader(flist):
    flist = pd.read_csv(flist, sep=' ', header=None).to_dict('list')
    return flist[0], flist[1]

def default_loader(path):
    return Image.open(path).convert('RGB')


