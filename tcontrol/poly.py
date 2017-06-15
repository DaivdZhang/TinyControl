import numpy as np

__all__ = ['conv', 'deconv', 'poly', 'roots']


def conv(*args):
    args = [np.array(i) for i in args]
    r = np.array([1])
    for i in args:
        r = np.convolve(i, r)
    return r


def deconv(u, v):
    return np.polydiv(u, v)


def poly(p):
    return np.poly(p)


def roots(p):
    """

    :param p:
    :type p: numpy.ndarray
    :return:
    :rtype: numpy.ndarray
    """
    return np.roots(p)
