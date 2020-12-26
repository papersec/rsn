
import numpy as np

def make_1darray(arr):
    """
    prevents numpy explore nested iterable class
    """

    ret = np.empty(len(arr), dtype=object)
    ret[:] = arr[:]
    return ret
