from tcontrol.lti import LinearTimeInvariant
from tcontrol.transferfunction import SISO
from tcontrol.pzmap import pzmap
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import widgets
from functools import partial, reduce
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
        fig, ax = plt.subplots()
        ax.axvline(x=0, color='black')
        ax.axhline(y=0, color='black')

        if 'xlim' in kwargs.keys():
            ax.set_xlim(*kwargs['xlim'])
        if 'ylim' in kwargs.keys():
            ax.set_ylim(*kwargs['xlim'])

        ax.plot(roots.real, roots.imag, color='red')
        p, z = pzmap(sys_, plot=False)
        ax.scatter(np.real(z), np.imag(z), s=50, marker='o', color='#069af3')
        ax.scatter(np.real(p), np.imag(p), s=50, marker='x', color='#fdaa48')
        ax.grid()
        plt.title('Root Locus')
        widgets.Cursor(ax, useblit=True, linewidth=2, linestyle='--')

        fig.canvas.mpl_connect("button_release_event", partial(_search_k, sys_=sys_))

        plt.show()

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


def _search_k(event, sys_):
    """

    :param event:
    :type event: matplotlib.backend_bases.MouseEvent
    :param sys_:
    :type sys_: SISO
    """
    s = complex(event.xdata, event.ydata)
    num = np.abs(sys_.pole() - s)
    den = np.abs(sys_.zero() - s)
    f = partial(reduce, lambda x, y: x*y)
    k = f(num)/f(den)
    print("K = {0:.5f} at {1:5f}{2:.5f}j".format(k, s.real, s.imag))
