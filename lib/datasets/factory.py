# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
import pdb
__sets = {}

import datasets.pascal_voc
import datasets.imagenet3d
import datasets.kitti
import datasets.kitti_tracking

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year, data_path=None:
                datasets.pascal_voc(split, year, data_path))

# KITTI dataset
for split in ['train', 'val', 'trainval', 'test']:
    name = 'kitti_{}'.format(split)
    __sets[name] = (lambda split=split, data_path=None:
            datasets.kitti(split))

# NTHU dataset
for split in ['71', '370']:
    name = 'nthu_{}'.format(split)
    __sets[name] = (lambda split=split, data_path=None:
            datasets.nthu(split, data_path))

pdb.set_trace()
def get_imdb(name, data_path=None):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    #pdb.set_trace()
    return __sets[name](data_path=data_path)

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
