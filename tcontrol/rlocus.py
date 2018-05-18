from tcontrol.transferfunction import ss2tf
from tcontrol.statespace import StateSpace
from tcontrol.pzmap import pzmap
from tcontrol.plot_utility import AnnotatedPoint, AttachedCursor
import numpy as np
from matplotlib import pyplot as plt

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
    :rtype: tuple(np.ndarray, np.ndarray)
    """
    if not sys_.issiso():
        raise NotImplementedError('rlocus is only for TransferFunction now.')
    if isinstance(sys_, StateSpace):
        system = ss2tf(sys_)
    else:
        system = sys_

    if kvect is None:
        nump = np.poly1d(system.num)
        denp = np.poly1d(system.den)

        d = _cal_multi_roots(nump, denp)
        k = -denp(d)/nump(d)
        k = k[np.where(k >= 0)]

        if k.dtype == np.complex:
            k = k[np.where(k.imag == 0)]
            k = np.real(k)
        k = np.sort(k)
        kvect = _setup_kvect(k)

    roots = _cal_roots(system, kvect)

    if plot:
        fig, ax = plt.subplots()
        ax.axvline(x=0, color='black')
        ax.axhline(y=0, color='black')

        if 'xlim' in kwargs.keys():
            ax.set_xlim(*kwargs['xlim'])
        if 'ylim' in kwargs.keys():
            ax.set_ylim(*kwargs['xlim'])

        line = []
        for r in roots:
            l, *_ = ax.plot(r.real, r.imag, picker=2)
            line.append(l)

        p, z = pzmap(system, plot=False)
        ax.scatter(np.real(z), np.imag(z), s=50, marker='o', color='#069af3')
        ax.scatter(np.real(p), np.imag(p), s=50, marker='x', color='#fdaa48')
        ax.grid()
        plt.title('Root Locus')

        cursor = RlocusAttachedCursor(ax, fig, line, sys_=system, linestyle='--')
        cursor.connect_event("pick_event", cursor)

        plt.show()
        cursor.disconnect()

    return roots, kvect


def _cal_roots(sys_, kvect):
    nump = np.poly1d(sys_.num)
    denp = np.poly1d(sys_.den)
    p = denp + kvect[0]*nump
    order = p.order
    roots = np.zeros((len(kvect), order), dtype=np.complex)
    for i, k in enumerate(kvect):
        p_ = denp + k*nump
        r = p_.roots
        r1 = r[r.imag == 0]
        r2 = r[r.imag != 0]
        roots[i] = np.append(r1, r2)
    return roots.T


def _cal_multi_roots(nump, denp):
    p = denp*np.polyder(nump) - np.polyder(denp)*nump
    return p.roots


def _setup_kvect(k):
    if len(k):
        kvect = np.linspace(0, k[0], 50)
        for i in range(1, len(k)):
            kvect = np.append(kvect, np.linspace(k[i - 1], k[i], 50))
        kvect = np.append(kvect, np.linspace(k[-1], k[-1]*50, 100))
        if kvect[-1] < 10:
            kvect = np.append(kvect, np.linspace(kvect[-1], kvect[-1] + 10, 10))
    else:
        kvect = np.linspace(0, 100, 10000)

    return kvect


class RlocusAnnotatedPoint(AnnotatedPoint):
    def __init__(self, ax, fig, sys_):
        super().__init__(ax, fig)
        self.ax.autoscale(False)
        self.sys_ = sys_
        self.anno.remove()
        self.anno = None

    def __call__(self, event):
        """

        :param event: A matplotlib event
        :type event: matplotlib.back_end.Event
        :return: None
        :rtype: None
        """
        if event.name == "button_press_event":
            self.handle_click(event)
            event.canvas.draw()
        else:
            return None

    def handle_click(self, event):
        """

        :param event: A matplotlib event
        :type event: matplotlib.back_end.Event
        :return: None
        :rtype: None
        """
        if event.inaxes is None:
            return None
        if self.anno is None:
            self.anno = self.init_annotate()

        s = complex(event.xdata, event.ydata)
        num = np.abs(self.sys_.pole() - s)
        den = np.abs(self.sys_.zero() - s)
        k = np.prod(num)/np.prod(den)

        if s.imag >= 0:
            text_str = "$K={0:.5f}$\n$s={1:.5f}+{2:.5f}j$".format(k, s.real, s.imag)
        else:
            text_str = "$K={0:.5f}$\n$s={1:.5f}{2:.5f}j$".format(k, s.real, s.imag)
        self.anno.xy = (event.xdata, event.ydata)
        self.anno.set_text(text_str)
        self.anno.set_x(event.xdata + 0.2)
        self.anno.set_y(event.ydata + 0.2)

        if self.dot is not None:
            self.dot.remove()
        self.dot = self.ax.scatter(event.xdata, event.ydata, marker='+', color='r', s=62)


class RlocusAttachedCursor(AttachedCursor):
    def __init__(self, ax, fig, line, *, sys_, **lineprops):
        super().__init__(ax, fig, **lineprops)
        self.ap = RlocusAnnotatedPoint(self.ax, self.ax.figure, sys_)
        self.line = line

    def __call__(self, event):
        if event.name == "pick_event":
            if event.artist not in self.line:
                return None
            self.ap(event.mouseevent)
        else:
            return None
