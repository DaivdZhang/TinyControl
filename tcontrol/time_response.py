# from tcontrol.lti import LinearTimeInvariant as LTI
from tcontrol.transferfunction import SISO, _siso_to_symbol
import numpy as np
import sympy as sym
from matplotlib import pyplot as plt
from functools import partial

__all__ = ['impulse', 'step', 'ramp', 'any_input']


def _get_cs(sys_, input_signal):
    """

    :param sys_:
    :type sys_: SISO
    :param input_signal:
    :type input_signal: str
    :return:
    :rtype: sym.Add
    """
    from sympy.parsing.sympy_parser import parse_expr

    s = sym.Symbol('s')
    t = sym.Symbol('t')

    input_expr = sym.simplify(parse_expr(input_signal))

    signal_table = {'step': 1/s, 'impulse': 1, '0': 0, 'ramp': 1/s**2,
                    'user': sym.laplace_transform(input_expr, t, s)[0]}
    gs, *_ = _siso_to_symbol(sys_.num, sys_.den)
    cs = gs*signal_table.get(input_signal, signal_table["user"])
    return cs


def _ilaplace(expr):
    """
    Usage:
        conduct the ilaplace transform

    :param expr: the expression
    :type expr: sympy.Add
    :return:
    :rtype:
    """
    from sympy.integrals.transforms import inverse_laplace_transform
    s = sym.Symbol('s')
    t = sym.Symbol('t')
    cs = expr.apart(s)

    tmp = sum(cs.args)
    if expr.equals(tmp):
        polys = [sym.nsimplify(i, tolerance=0.001, rational=True) for i in tmp.args]
        ilaplace_p = partial(inverse_laplace_transform, s=s, t=t)
        ct = 0
        for i in polys:
            i = ilaplace_p(i)
            ct += i
    else:
        cs = sym.nsimplify(expr, tolerance=0.001, rational=True)
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


def step(sys_, t=None, *, plot=True):
    if t is None:
        t = np.linspace(0, 10, 1000)

    y, _ = _any_input(sys_, t, 'step')

    if plot:
        _plot_curve(y, t, 'Step Response')

    return y, t


def impulse(sys_, t=None, *, plot=True):
    if t is None:
        t = np.linspace(0, 10, 1000)

    y, _ = _any_input(sys_, t, 'impulse')

    if plot:
        _plot_curve(y, t, 'Impulse Response')

    return y, t


def ramp(sys_, t=None, *, plot=True):
    if t is None:
        t = np.linspace(0, 10, 1000)

    y, _ = _any_input(sys_, t, 'ramp')

    if plot:
        _plot_curve(y, t, 'Ramp Response')

    return y, t


def any_input(sys_, t=None, input_signal=None, *, plot=True):
    """
    Usage:
        calculate the output with any input

    Example:
        >>> import tcontrol as tc
        >>> system = tc.tf([1], [1, 0, 1])
        >>> tc.any_input(system, None, "t**2 + 3")
        >>> tc.plot.show()

    :param sys_: the transfer function of the system
    :type sys_: SISO
    :param t: time
    :type t: np.ndarray | None
    :param input_signal:
    :type input_signal: str
    :param plot: if plot is true it will draw the picture
    :type plot: bool
    :return: output value and time
    :rtype: (np.ndarray, np.ndarray)
    """
    if t is None:
        t = np.linspace(0, 10, 1000)

    y, _ = _any_input(sys_, t, input_signal)

    if plot:
        _plot_curve(y, t, input_signal)

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
