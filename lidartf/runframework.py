# -*- coding: utf-8 -*-
"""
Module to perform the framework testing.

@author: Matheus Boni Vicari (2017).
"""

import numpy as np
import imp
import os
import pandas as pd
import sys
from tests.wlseparation import test_separation


def run(fun, results_folder, plot_cloud, dataset, *args):

    """
    Function to run the testing framework. This is the main function in the
    package and manages input, testing and output.

    Parameters
    ----------
    fun: list of str
        List containing module path and function name (in this order) to import
        and test.
    results_folder: str
        Path of the directory to save the testing results. Path must finish
        with a separator (e.g. / or \).
    plot_cloud: bool
        Option to plot or not the separated point clouds.
    dataset: list
        List of dataset paths to use as data for the testing.
    args: list
        List of arguments necessary to run the function to be tested.
        These arguments should be inserted in the same order as required by
        function to be tested. Even if using only single values for each
        argument, it should be inserted inside a list.

    Returns
    -------
    res: pandas.DataFrame
        Set of results for the testing of the function.
    params: list
        List of parameters used to run the tested function.

    Usage
    -----
    >>> dataset = ['path/to/dataset.txt']
    >>> fun = ['path/to/module_file', 'function_name']
    >>> res, t = run(fun, 'results/', True, dataset, [10, 20, 40, 100],\
 [100, 200, 300])
    """

    # Importing module 'm' and function 'f' to test.
    m = import_(fun[0])
    f = getattr(m, fun[1])

    # Initializing empty lists to store parameters and results of the test.
    results = []
    params = []

    # Looping over every data in dataset.
    for i in dataset:

        # Importing current dataset.
        data = np.loadtxt(i, delimiter=' ')

        # Extracting the dataset filename.
        filename = os.path.basename(i).split('.')[0]

        # Running the separation test.
        results, params = test_separation(data, f, filename, results_folder,
                                          plot_cloud, *args)

    # Creating a pandas.DataFrame from the test results.
    res = pd.DataFrame(results, columns=['accuracy', 'tp', 'fp', 'tn', 'fn',
                                         'F_wood', 'F_leaf', 'k', 'time'])

    # Saving the testing parameters to a text file.
    np.savetxt(results_folder + 'params_' + filename + '.txt', params,
               fmt='%1.2f')

    # Saving the testing results to a text file.
    res.to_csv(results_folder + 'results_' + filename + '.txt',
               float_format='%1.2f')

    return res, params


def import_(filename):

    """
    Function to import a Python module from a filename.

    Parameters
    ----------
    filename: str
        Path of the module to import without extension.

    """

    # Splitting filename to extract path and name of the module.
    path, name = os.path.split(filename)
    # Extracting name and extension.
    name, ext = os.path.splitext(name)

    # Appending modulo path to system path.
    sys.path.append(path)

    # Finding module and importing it.
    file_, filename, data = imp.find_module(name, [path])
    mod = imp.load_module(name, file_, filename, data)

    return mod


if __name__ == "__main__":

    fun = [r'path_to_module', 'name_of_function_to_test']
    dataset = [r'data_to_use_in_test_1',
               r'data_to_use_in_test_1']
    res, t = run(fun, 'path_to_results', True, dataset, [list_arguments_1], [list_of_arguments_2])
