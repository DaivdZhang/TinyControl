from tcontrol.lti import LinearTimeInvariant
from tcontrol.transferfunction import SISO
from tcontrol.pzmap import pzmap
import numpy as np
from matplotlib import pyplot as plt
from functools import partial
import operator

__all__ = ["rlocus"]


def rlocus(sys_, kvect=None, *, plot=True, **kwargs):
    """
    Usage:
        the root locus of the system

    Example:
        >>> import tcontrol as tc
        >>> import numpy as np
        >>> system = tc.tf([1], [1, 2, 1])
        >>> tc.rlocus(system, np.linspace(0, 5, 1000))
        >>> tc.plot.show()

    :param sys_: the transfer function of the system
    :type sys_: SISO | LTI
    :param kvect: k from 0 to inf
    :type kvect: np.ndarray
    :param plot: if plot is true it will draw the picture
    :type plot: bool
    :return: roots of the den and kvect
    :rtype: (np.ndarray, np.ndarray)
    """
    if not isinstance(sys_, SISO) and isinstance(sys_, LinearTimeInvariant):
        raise NotImplementedError('rlocus is only for SISO system now')

    ol_gains = np.linspace(0, 100, 10000) if kvect is None else kvect

    roots = _cal_roots(sys_, ol_gains)
    roots = _sort_roots(roots)

    if plot:
        plt.axvline(x=0, color='black')
        plt.axhline(y=0, color='black')
        if 'xlim' in kwargs.keys():
            plt.xlim(*kwargs['xlim'])
        if 'ylim' in kwargs.keys():
            plt.ylim(*kwargs['ylim'])

        plt.plot(roots.real, roots.imag, color='red')
        p, z = pzmap(sys_, plot=False)
        plt.scatter(np.real(z), np.imag(z), s=50, marker='o', color='#069af3')
        plt.scatter(np.real(p), np.imag(p), s=50, marker='x', color='#fdaa48')
        plt.grid()
        plt.title('Root Locus')
        plt.draw()

    return roots, kvect


def _cal_roots(sys_, kvect):
    def _cal_k_roots(k, nump, denp):
        p_ = denp + k*nump
        r = np.roots(p_)
        return np.sort(r)

    cal_roots_p = partial(_cal_k_roots, nump=np.poly1d(sys_.num), denp=np.poly1d(sys_.den))
    roots = tuple(map(cal_roots_p, kvect))
    roots = np.array(roots)
    return roots


def _sort_roots(roots):
    sorted_ = np.zeros_like(roots)
    sorted_[0] = roots[0]
    pre_row = sorted_[0]
    for n, row in enumerate(roots[1:, :]):
        distance_arr = [np.abs(i - pre_row) for i in row]
        _ = [i.argmin() for i in distance_arr]
        _ = sorted(zip(_, row), key=operator.itemgetter(0))
        _ = [i[-1] for i in _]
        sorted_[n + 1] = np.array(_)
        pre_row = sorted_[n + 1, :]
    return sorted_
