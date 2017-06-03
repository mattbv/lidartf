# -*- coding: utf-8 -*-
"""
Module to run an intersection analysis between two point clouds.

@author: Matheus Boni Vicari (2017).
"""

import numpy as np
import pandas as pd


def get_diff(arr1, arr2):

    """
    Function to generate a difference point cloud (points not intersected)
    between point clouds.

    Parameters
    ----------
    arr1: numpy.ndarray
        First point cloud to analyze.

    arr2: numpy.ndarray
        Second point cloud to analyze.

    Returns
    -------
    diff: numpy.ndarray
        Difference point cloud.

    Examples
    --------
    >>> arr1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    >>> arr1
    array([[0, 0, 0],
           [1, 1, 1],
           [2, 2, 2]])
    >>> arr2 = np.array([[0, 0, 0], [1, 1, 1], [3, 3, 3]])
    >>> arr2
    array([[0, 0, 0],
           [1, 1, 1],
           [3, 3, 3]])
    >>> get_diff(arr1, arr2)
    array([[2, 2, 2],
           [3, 3, 3]])

    """

    # Making sure arr1 and arr2 have the same number of dimensions.
    assert arr1.shape[1] == arr2.shape[1]

    # Stacking both arrays.
    arr3 = np.vstack((arr1, arr2))

    # Generating a pandas.DataFrame from the stacked array.
    df = pd.DataFrame(arr3)

    # Removing all points (rows) that are not unique.
    diff = df.drop_duplicates(keep=False)

    return np.asarray(diff)


def count_intersection(arr1, arr2):

    """
    Function to calculate the number of common points between two clouds.

    Parameters
    ----------
    arr1: numpy.ndarray
        First point cloud to analyze.

    arr2: numpy.ndarray
        Second point cloud to analyze.

    Returns
    -------
    count: int
        Number of common points.

    Examples
    --------
    >>> arr1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    >>> arr1
    array([[0, 0, 0],
           [1, 1, 1],
           [2, 2, 2]])
    >>> arr2 = np.array([[0, 0, 0], [1, 1, 1], [3, 3, 3]])
    >>> arr2
    array([[0, 0, 0],
           [1, 1, 1],
           [3, 3, 3]])
    >>> count_intersection(arr1, arr2)
    2

    """

    # Making sure arr1 and arr2 have the same number of dimensions.
    assert arr1.shape[1] == arr2.shape[1]

    # Stacking both arrays.
    arr3 = np.vstack((arr1, arr2))

    # Generating a pandas.DataFrame from the stacked array.
    df = pd.DataFrame(arr3)

    # Obtainin the duplicated points in the DataFrame.
    diff = np.asarray(df.duplicated(keep=False))

    # Calculating the number of intersected points.
    intercount = np.sum(diff) / 2

    return intercount.astype(np.int)
