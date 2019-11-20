import numpy as np


class DAL_Utils:
    @staticmethod
    def numpy_load(path, allow_pickle=False):
        return np.load(path, allow_pickle=allow_pickle)
