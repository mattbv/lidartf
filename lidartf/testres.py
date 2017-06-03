# -*- coding: utf-8 -*-
"""
Module to perform the results assessment of the framework.

@author: Matheus Boni Vicari (2017).
"""

from __future__ import division
from utils.intersection import count_intersection


# Defining the summary function.
def summary(Cw, Cl, W, L):

    """
    Function to generate a summary of all result parameters calculated within
    this module.

    Parameters
    ----------
    R: tuple
            Sets of 3D point coordinates from the points cloud output by the
            tested function.
    T: tuple
            Sets of 3D point coordinates, referenced as truth points, from the
            input test data.
    F: tuple
            Sets of 3D point coordinates, referenced as false points, from the
            input test data.

    Returns
    -------
    p_removed: int
            Total amount of points not present in the result set of points.
            This parameter might be helpful when assessing a classification or
            filtering processing.
    accuracy: float
            Accuracy of the processing when observing the tests of true/false
            positives and negatives.
    tp: numpy.float64
            Number of true positives.
    fp: numpy.float64
            Number of false positives.
    tn: numpy.float64
            Number of true negatives.
    fn: numpy.float64
            Number of false negatives.
    Fscore: numpy.float64
            F-score calculated value.
    k: numpy.float64
            Cohen's kappa.

    """

    tp, fp, tn, fn = test_results(Cw, Cl, W, L)

    # Executing test_results.
#    tp, fp, tn, fn = test_results(Cw, Cl, W, L)

    # Calculating the total amount of points processed.
    total = tp + fp + tn + fn

    # Calculating the accuracy of the processing. This is done by the ratio
    # of the total amount of points correctly processed (summation of true
    # positives and true negatives) and the total amount of points processed.
    accuracy = (tp + tn) / total

    # Executing the fscore and kappa functions.
    F_wood = fscore(tp, fp, fn)
    F_leaf = fscore(tn, fn, fp)
    k = kappa(tp, fp, tn, fn)

    return accuracy, tp, fp, tn, fn, F_wood, F_leaf, k


# Defining the fscore function.
def fscore(tp, fp, fn):

    """
    This function calculates the F-score of a set of results from the testing
    framework.
    The calculations area based on Goutte and Gaussier (2005), Sokolova et al.
    (2006) and Tao et al. (2015).

    Parameters
    ----------
    R: tuple
            Sets of 3D point coordinates from the points cloud output by the
            tested function.
    T: tuple
            Sets of 3D point coordinates, referenced as truth points, from the
            input test data.
    F: tuple
            Sets of 3D point coordinates, referenced as false points, from the
            input test data.

    Returns
    -------
    Fscore: numpy.float64
            F-score calculated value.

    References
    ----------
    .. [1] Goutte, C., Gaussier, E., 2005. A probabilistic interpretation
           of precision, recall and F-score, with implication for evaluation.
           Lect. Notes Comput. Sci. 3408, 345–359.

    .. [2] Sokolova, M., Japkowicz, N., Szpakowicz, S., 2006. Beyond accuracy,
           F-score and ROC: a family of discriminant measures for performance
           evaluation. In: Sattar, A., Kang, B.-H. (Eds.), AI 2006: Advances in
           Artificial Intelligence. Springer, Berlin, Heidelberg, pp.
           1015–1021.

    .. [3] Tao, S., Wu, F., Guo, Q., Wang, Y., Li, W., Xue, B., Hu, X., Li, P.,
           Tian, D., Li, C., Yao, H., Li, Y., Xu, G., Fang, J., 2015.
           Segmenting tree crowns from terrestrial and mobile LiDAR data by
           exploring ecological theories. ISPRS Journal of Photogrammetry and
           Remote Sensing 110, 66–76.

    """

    # Calculating recall (r) and precision (p).
    r = tp / (tp + fn)
    p = tp / (tp + fp)

    # Calculating the F-score.
    Fscore = 2 * ((r * p) / (r + p))

    return Fscore


# Defining the kappa function.
def kappa(tp, fp, tn, fn):

    """
    This function calculates kappa according to the original publication from
    Jacob Cohen (1960). The only modification made for the use of Cohen's kappa
    on the testing framework is the if-else control to avoid division by 0 when
    the test data has no leaf/noise points. In these cases, the k variable is
    set to a fill value (9999).

    Parameters
    ----------
    R: tuple
            Sets of 3D point coordinates from the points cloud output by the
            tested function.
    T: tuple
            Sets of 3D point coordinates, referenced as truth points, from the
            input test data.
    F: tuple
            Sets of 3D point coordinates, referenced as false points, from the
            input test data.

    Returns
    -------
    k: numpy.float64
            Cohen's kappa.

    References
    ----------
    .. [1] Cohen, J. A Coefficient of Agreement for Nominal Scales. Educational
           and Psychological Measurement, April 1960, 20: 37-46.

    """

    # Executing the functions p_proportionate_agreement and
    # p_random_agreement to obtain the po and pe, respectively.
    po = p_proportionate_agreement(tp, fp, tn, fn)
    pe = p_random_agreement(tp, fp, tn, fn)

    # Testing if pe is larger than 1. If so, calculate k and,
    # if not, assign a fill value to k (9999).
    if pe < 1:
        k = (po - pe) / (1 - pe)
    else:
        k = 9999
        return k

    return abs(k)


# Defining the p_proportionate_agreement function.
def p_proportionate_agreement(tp, fp, tn, fn):

    """
    The probability of proportionate agreement is the relative amount
    of points in agreement with the "truth", which is the sum of true
    positives and false positives divided by the total number of points

    Parameters
    ----------
    tp: int or float
            Number of true positives.
    fp: int or float
            Number of false positives.
    tn: int or float
            Number of true negatives.
    fn: int or float
            Number of false negatives.

    Returns
    -------
    po: numpy.float64
        probability of proportionate agreement.

    """

    # Calculating the total amount of points.
    total = tp + fp + tn + fn

    # Calculating the proportionate agreement.
    po = (tp + tn) / total
    return po


# Defining the p_random_agreement function.
def p_random_agreement(tp, fp, tn, fn):

    """
    The probability of random agreement is the probability of the "truth"
    to be achieved randomly, independent of the method used to classify/
    identify each point as wood or leaf/noise.

    Parameters
    ----------
    tp: int or float
            Number of true positives.
    fp: int or float
            Number of false positives.
    tn: int or float
            Number of true negatives.
    fn: int or float
            Number of false negatives.

    Returns
    -------
    pe: numpy.float64
            Probability of random agreement.

    """

    # Calculating the total amount of points.
    total = tp + fp + tn + fn

    m_original = ((tp + fn) * (tp + fp)) / total
    m_classification = ((fp + tn) * (fn + tn)) / total

    # Calculating the probability of random agreement.
    pe = (m_original + m_classification) / total

    return pe


# Defining the function test_results function.
def test_results(Cw, Cl, W, L):

#    tw = count_intersection(Cw, W)
#    fw = count_intersection(Cw, L)
#    tl = count_intersection(Cl, L)
#    fl = count_intersection(Cl, W)

    # OR

    tw = count_intersection(Cw, W)
    tl = count_intersection(Cl, L)
    fw = abs(Cw.shape[0] - tw)
    fl = abs(Cl.shape[0] - tl)

    return tw, fw, tl, fl


if __name__ == "__main__":
    pass
