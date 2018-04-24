import numpy as np
#from tqdm import trange
#import builtins
#builtins.range=trange

from _context import dl_uncertainty
from dl_uncertainty.data.datasets import load_voc2012_segmentation
from dl_uncertainty.dirs import DATASETS
import dl_uncertainty.visualization as vis


voc = load_voc2012_segmentation(f"{DATASETS}/VOC2012", 'train#')
assert voc.images[0].shape[0] == 500
print(np.unique(voc.labels))

vis.view_semantic_segmentation(voc)

