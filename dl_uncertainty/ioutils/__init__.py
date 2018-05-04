import builtins
from tqdm import tqdm, trange

class RangeProgressBar():

    def __enter__(self):
        self.range = builtins.range

        def progress_range(*args, **kwargs):
            return tqdm(self.range(*args, **kwargs), leave=True)

        builtins.range = progress_range

    def __exit__(self, *args):
        builtins.range = self.range
