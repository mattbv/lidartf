# -*- coding: utf-8 -*-
"""
Module to perform the wood-leaf separation testing.

@author: Matheus Boni Vicari (2017).
"""
import numpy as np
import itertools
import testres as tr
import pandas as pd
import time
import mayavi.mlab as mlab


def test_separation(data, f, filename, results_folder, plot_cloud, *args):

    # Initializing empty lists to store parameters and results of the test.
    results = []
    params = []

    # Generating the reference wood and leaf datasets.
    wood = remove_duplicates(data[data[:, 3] == 0])
    leaf = remove_duplicates(data[data[:, 3] == 1])

    # Stacking references data to create the testing dataset.
    test_data = np.vstack((wood, leaf))

    # Iterating over the product of all input arguments. This will generate
    # all possible combinations of arguments from 'args' to test.
    for j in itertools.product(*args):

        # Starting the time counter.
        start = time.time()

        # Trying to separate the testing dataset with the current
        # arguments.
        try:
            # Run function 'f' with arguments j.
            w_out, l_out, p = f(test_data, *j)

            # If set, plot and save separated point clouds.
            if plot_cloud:
                mlab.figure(bgcolor=(1, 1, 1))
                mlab.points3d(w_out[:, 0], w_out[:, 1], w_out[:, 2],
                              color=(0.4, 0.2, 0), mode='point')
                mlab.points3d(l_out[:, 0], l_out[:, 1], l_out[:, 2],
                              color=(0, 0.4, 0), mode='point')
                mlab.savefig(results_folder + 'cloud_' + filename + '_' +
                             '_'.join(map(str, j)) + '.png',
                             size=[1920, 1080])
                mlab.close()

            # Testing the separated point clouds against the reference
            # point clouds.
            tempres = tr.summary(w_out, l_out, wood[:], leaf[:])
            # Joining processing time to the testing results.
            tempres = tempres + (time.time() - start, )

            # Printing current iteration results.
            print('\n Current results for dataset %s and arguments %s:' %
                  (filename, j))
            print tempres

            # Appending the test results to results list.
            results.append(tempres[:])

            # Deleting separated point clouds and current results to avoid
            # conflicts when
            del(w_out, l_out, tempres)

        except:
            # If not possible to separate the point cloud, fill current results
            # as zeroes.
            results.append((0, 0, 0, 0, 0, 0, 0, 0, 0))

        # Append current parameters (arguments).
        params.append(j)

    return results, params


def remove_duplicates(arr):

    """
    Function to remove duplicate rows from an array.

    Parameters
    ----------
    arr: numpy.ndarray
        N-dimensional array to uniquify rows.

    Returns
    -------
    unique: numpy.ndarray
        Array with unique rows.

    """

    # Creating a pandas.DataFrame from the input array.
    df = pd.DataFrame({'x': arr[:, 0], 'y': arr[:, 1], 'z': arr[:, 2]})

    # Removing duplicate rows.
    unique = df.drop_duplicates(['x', 'y', 'z'])

    return np.asarray(unique)
