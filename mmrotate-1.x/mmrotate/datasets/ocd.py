# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
from typing import List

from .dota import DOTADataset
from mmrotate.registry import DATASETS


@DATASETS.register_module()
class OCDDataset(DOTADataset):

    METAINFO = {
        'classes':
        ('NormalCell', 'RoundCell'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42), (189, 183, 107)]
    }

@DATASETS.register_module()
class OCDClassAgnosticDataset(OCDDataset):

    METAINFO = {
        'classes':
        ('NormalCell',),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42),]
    }
