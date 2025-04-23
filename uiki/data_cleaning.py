import numpy as np

def clean_data(intervals):
    """
    Remove PP intervals outside the interquartile range.
    intervals: array-like of keystroke intervals
    returns: numpy array of cleaned intervals
    """
    arr = np.array(intervals)
    if arr.size == 0:
        return arr
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    return arr[(arr >= q1) & (arr <= q3)]