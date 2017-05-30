from src.lti import LinearTimeInvariant as LTI
from src.transferfunction import SISO
from src.pzmap import pzmap
import numpy as np
from matplotlib import pyplot as plt
from functools import partial

__all__ = ["rlocus"]


def rlocus(sys_, kvect=None, *, plot=True, **kwargs):
    """

    :param kvect:
    :type kvect: array like
    :param sys_:
    :type sys_: SISO
    :param plot:
    :type plot: bool
    :return:
    :rtype:
    """
    num, den = np.poly1d(sys_.num), np.poly1d(sys_.den)
    ol_gains = np.linspace(0, 100, 10000) if kvect is None else kvect

    def cal_roots(k, nump, denp):
        p_ = denp + k*nump
        r = np.roots(p_)
        return np.sort(r)

    cal_roots_p = partial(cal_roots, nump=num, denp=den)
    roots = list(map(cal_roots_p, ol_gains))
    roots = np.array(roots)

    roots = _sort_roots(roots)
    roots = np.vstack(roots)

    if plot:
        plt.axvline(x=0, color='black')
        plt.axhline(y=0, color='black')
        try:
            plt.xlim(*kwargs['xlim'])
        except KeyError:
            pass
        plt.plot(roots.real, roots.imag, color='red')
        p, z = pzmap(sys_, plot=False)
        plt.scatter(np.real(z), np.imag(z), s=50, marker='o', color='#069af3')
        plt.scatter(np.real(p), np.imag(p), s=50, marker='x', color='#fdaa48')
        plt.grid()
        plt.title('Root Locus')
        plt.draw()


def _sort_roots(roots):
    sorted_ = np.zeros_like(roots)
    sorted_[0] = roots[0]
    pre_row = sorted_[0]
    for n, row in enumerate(roots[1:, :]):
        n += 1
        _ = [np.abs(i - pre_row) for i in row]
        _ = np.array([i.argmin() for i in _])
        pi = -1
        for i, e in zip(_, row):
            if pi == i:
                i = i - 1 if row.shape[0] - 1 == i else i + 1
            sorted_[n][i] = e
            pi = i
        pre_row = sorted_[n, :]
    return sorted_
