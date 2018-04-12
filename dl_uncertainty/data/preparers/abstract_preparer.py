import abc


class AbstractPreparer(object):
    @staticmethod
    @abc.abstractmethod
    def prepare(data_path: str) -> str:
        """
        Converts a raw dataset at <data_path>/* into
            1) <data_path>.prepared/images/*.png
                - standardized (any, but equal) dimensions (trimmed and filled if necessary)
                - RGB, bytes
            2) <data_path>.prepared/labels/*.label
                - 2D byte array with dimensions (height, width) like images
                - 0-valued bytes represent unknown, nothing or 'don't care'
                - 1...n-valued bytes represent classes
            3) <data_path>.prepared/info.cfg
                - contains the number of different classes (n) (including 'don't care')
        data.dataset_dir.py functions.
        """
        out_data_path = data_path + '.prepared'
        return out_data_path
