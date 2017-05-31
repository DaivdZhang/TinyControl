from .lti import LinearTimeInvariant as LTI
from .transferfunction import SISO, _siso_to_symbol
import numpy as np
import sympy as sym
from matplotlib import pyplot as plt

__all__ = ['impulse', 'step', 'ramp']


def _get_cs(sys_, input_signal):
    """

    :param sys_:
    :type sys_: SISO
    :param input_signal:
    :type input_signal: str
    :return:
    :rtype: sym.Add
    """
    s = sym.Symbol('s')
    signal_table = {'step': 1/s, 'impulse': 1, '0': 0, 'ramp': 1/s**2}
    gs, *_ = _siso_to_symbol(sys_.num, sys_.den)
    cs = gs*signal_table[input_signal]
    return cs


def _ilaplace(expr):
    """

    :param expr:
    :type expr: sym.Add
    :return:
    :rtype:
    """
    from sympy.integrals.transforms import inverse_laplace_transform
    s = sym.Symbol('s')
    t = sym.Symbol('t')
    cs = sym.nsimplify(expr, tolerance=0.001, rational=True)
    cs = cs.apart(s)

    tmp = 0
    for i in cs.args:
        tmp += i
    if expr.equals(tmp):
        ct = 0
        for i in tmp.args:
            ct += inverse_laplace_transform(sym.nsimplify(i, tolerance=0.01, rational=True), s, t)
    else:
        ct = inverse_laplace_transform(cs, s, t)

    if ct.has(sym.Heaviside):
        ct = ct.replace(sym.Heaviside, sym.numbers.One)
    if ct.has(sym.InverseLaplaceTransform):
        ct = ct.replace(sym.InverseLaplaceTransform, sym.numbers.Zero)

    return ct


def _any_input(sys_, t, input_signal='0'):
    t_ = sym.Symbol('t')
    output = _get_cs(sys_, input_signal)
    output = _ilaplace(output)
    output_func = sym.lambdify(t_, output, modules=['numpy'])
    y = output_func(t)
    return y, t


def step(sys_, t=None, plot=True):
    if t is None:
        t = np.linspace(0, 10, 1000)

    y, _ = _any_input(sys_, t, 'step')

    if plot:
        _plot_curve(y, t, 'Step Response')

    return y, t


def impulse(sys_, t=None, plot=True):
    if t is None:
        t = np.linspace(0, 10, 1000)

    y, _ = _any_input(sys_, t, 'impulse')

    if plot:
        _plot_curve(y, t, 'Impulse Response')

    return y, t


def ramp(sys_, t=None, plot=True):
    if t is None:
        t = np.linspace(0, 10, 1000)

    y, _ = _any_input(sys_, t, 'ramp')

    if plot:
        _plot_curve(y, t, 'Ramp Response')

    return y, t


def _plot_curve(y, t, title):
    plt.title(title)
    plt.xlabel('t/s')
    plt.ylabel('Amplitude')
    plt.axvline(x=0, color='black')
    plt.axhline(y=0, color='black')
    plt.plot(t, y)
    plt.grid()
    plt.draw()
