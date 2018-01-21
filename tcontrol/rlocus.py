from tcontrol.lti import LinearTimeInvariant
from tcontrol.transferfunction import TransferFunction
from tcontrol.pzmap import pzmap
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import widgets
from functools import partial, reduce

__all__ = ["rlocus"]


def rlocus(sys_, kvect=None, *, plot=True, **kwargs):
    """
    Use:
        the root locus of the system

    Example:
        >>> import tcontrol as tc
        >>> import numpy as np
        >>> system = tc.tf([1], [1, 2, 1])
        >>> tc.rlocus(system, np.linspace(0, 5, 1000))

    :param sys_: the transfer function of the system
    :type sys_: SISO | LTI
    :param kvect: k from 0 to inf
    :type kvect: np.ndarray
    :param plot: if plot is true it will draw the picture
    :type plot: bool
    :return: roots of the den and kvect
    :rtype: (np.ndarray, np.ndarray)
    """
    if not isinstance(sys_, TransferFunction) and isinstance(sys_, LinearTimeInvariant):
        raise NotImplementedError('rlocus is only for TransferFunction now.')

    nump = np.poly1d(sys_.num)
    denp = np.poly1d(sys_.den)
    if kvect is None:
        d = _cal_multi_roots(nump, denp)
        k = -denp(d)/nump(d)
        k = k[np.where(k >= 0)]
        if k.dtype == complex:
            k = k[np.where(k.imag == 0)]
            k = np.real(k)
        k = np.sort(k)

        kvect = np.linspace(0, k[0], 100)
        for i in range(1, len(k)):
            kvect = np.append(kvect, np.linspace(k[i - 1], k[i], 100))
        kvect = np.append(kvect, np.linspace(k[-1], k[-1]*50, 500))

    ol_gains = kvect
    roots = _cal_roots(sys_, ol_gains)

    if plot:
        fig, ax = plt.subplots()
        ax.axvline(x=0, color='black')
        ax.axhline(y=0, color='black')

        if 'xlim' in kwargs.keys():
            ax.set_xlim(*kwargs['xlim'])
        if 'ylim' in kwargs.keys():
            ax.set_ylim(*kwargs['xlim'])

        for r in roots:
            ax.plot(r.real, r.imag)

        p, z = pzmap(sys_, plot=False)
        ax.scatter(np.real(z), np.imag(z), s=50, marker='o', color='#069af3')
        ax.scatter(np.real(p), np.imag(p), s=50, marker='x', color='#fdaa48')
        ax.grid()
        plt.title('Root Locus')
        cursor = widgets.Cursor(ax, useblit=True, linewidth=1, linestyle='--')

        fig.canvas.mpl_connect("button_release_event", partial(_search_k, sys_=sys_))

        plt.show()

    return roots, kvect


def _cal_roots(sys_, kvect):
    nump = np.poly1d(sys_.num)
    denp = np.poly1d(sys_.den)
    p = denp + kvect[0]*nump
    order = p.order
    roots = np.zeros((len(kvect), order), dtype=complex)
    for i, k in enumerate(kvect):
        p_ = denp + k*nump
        roots[i] = p_.roots

    roots = np.sort(roots, axis=1)
    return roots.T


def _cal_multi_roots(nump, denp):
    p = denp*np.polyder(nump) - np.polyder(denp)*nump
    return p.roots


# def _sort_roots(roots):
#     sorted_ = np.zeros(roots.shape, dtype=complex)
#     sorted_[0] = roots[0]
#     pre_row = sorted_[0]
#     for n, row in enumerate(roots[1:, :]):
#         distance_arr = [np.abs(i - pre_row) for i in row]
#         _ = [i.argmin() for i in distance_arr]
#         _ = sorted(zip(_, row), key=operator.itemgetter(0))
#         _ = [i[-1] for i in _]
#         sorted_[n + 1] = np.array(_)
#         pre_row = sorted_[n + 1, :]
#     return sorted_


def _search_k(event, sys_):
    """

    :param event:
    :type event: matplotlib.backend_bases.MouseEvent
    :param sys_:
    :type sys_: TransferFunction
    """
    s = complex(event.xdata, event.ydata)
    num = np.abs(sys_.pole() - s)
    den = np.abs(sys_.zero() - s)
    f = partial(reduce, lambda x, y: x*y)
    k = f(num)/f(den)
    if s.imag >= 0:
        print("K = {0:.5f} at {1:.5f}+{2:.5f}j".format(k, s.real, s.imag))
    else:
        print("K = {0:.5f} at {1:.5f}{2:.5f}j".format(k, s.real, s.imag))


if __name__ == "__main__":
    import timeit

    timer = timeit.Timer(
        "rlocus(system, np.linspace(0, 100, 10000), xlim=[-5, 0.5], plot=False)",
        "from tcontrol import tf, rlocus; system = tf([0.5, 1], [0.5, 1, 1]);"
        "import numpy as np")
    r_ = timer.repeat(3, 5)
    print("{0:.3f} ms\n".format(sum(r_)/15*1000))
