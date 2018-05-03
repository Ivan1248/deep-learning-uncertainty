# Old
from .dataset import Dataset, DatasetGenerator, MiniBatchReader, save_dataset, load_dataset
# New
from torch.utils.data.dataset import ConcatDataset
from .dataset_loaders import AbstractDataset, DataLoader