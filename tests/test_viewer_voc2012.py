import numpy as np
#from tqdm import trange
#import builtins
#builtins.range=trange

from _context import dl_uncertainty
from dl_uncertainty.data.datasets import load_voc2012_segmentation
from dl_uncertainty.dirs import DATASETS
from dl_uncertainty.utils.visualization import view_semantic_segmentation

voc = load_voc2012_segmentation(f"{DATASETS}/VOC2012", 'train')
assert voc.images[0].shape[0] == 500
print(np.unique(voc.labels))

view_semantic_segmentation(voc)
