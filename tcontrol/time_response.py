# from tcontrol.lti import LinearTimeInvariant as LTI
from tcontrol.transferfunction import SISO, _siso_to_symbol
from tcontrol.statespace import StateSpace, continuous_to_discrete, tf2ss
import numpy as np
import sympy as sym
from functools import partial
from tcontrol.plot_func import plot_response_curve
import warnings

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
    Use:
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


def __any_input(sys_, t, input_signal='0'):
    t_ = sym.Symbol('t')
    output = _get_cs(sys_, input_signal)
    output = _ilaplace(output)
    output_func = sym.lambdify(t_, output, modules=['numpy'])
    y = output_func(t)
    return y, t


def __step(sys_, t=None, *, plot=True):
    if t is None:
        t = np.linspace(0, 10, 1000)

    y, _ = __any_input(sys_, t, 'step')

    if plot:
        plot_response_curve(y, t, 'Step Response')

    return y, t


def __impulse(sys_, t=None, *, plot=True):
    if t is None:
        t = np.linspace(0, 10, 1000)

    y, _ = __any_input(sys_, t, 'impulse')

    if plot:
        plot_response_curve(y, t, 'Impulse Response')

    return y, t


def __ramp(sys_, t=None, *, plot=True):
    if t is None:
        t = np.linspace(0, 10, 1000)

    y, _ = __any_input(sys_, t, 'ramp')

    if plot:
        plot_response_curve(y, t, 'Ramp Response')

    return y, t


# convert system to state space then get result
def _any_input(sys_, t, input_signal=0, init_cond=None):
    """

    :param sys_: the system to be calculated
    :type sys_: SISO
    :param t: time
    :type t: None | np.ndarray
    :param input_signal: a array contain the input signal
    :type input_signal: int | float | np.ndarray | tuple(np.ndarray)
    :param init_cond: initial condition
    :type init_cond: None | int | float | np.ndarray
    :return: output and time
    :rtype:
    """
    # convert transfer function or continuous system to discrete system
    dt = t[1] - t[0]
    if dt > 0.02:
        warnings.warn("sample time is big than 0.02s, which will lead to low accuracy",
                      stacklevel=3)
    if sys_.issiso():
        d_sys_ = continuous_to_discrete(tf2ss(sys_), dt)
    elif isinstance(sys_, StateSpace):
        if sys_.isctime():
            d_sys_ = continuous_to_discrete(sys_, dt)
        else:
            d_sys_ = sys_
    else:
        raise TypeError("wrong type of arg")

    # check the input_signal validity
    if isinstance(input_signal, (int, float)):
        u = np.repeat(input_signal, len(t))
    elif isinstance(input_signal, np.ndarray):
        u = input_signal
    elif d_sys_.inputs > 1 and isinstance(input_signal, tuple):
        raise NotImplemented("not support multi inputs right row")  # TODO: complete this
    else:
        raise TypeError("unexpected type of input_signal")

    if init_cond is None:
        init_cond = np.mat(np.zeros((d_sys_.A.shape[0], 1)))
    else:
        # check the shape of init_cond
        if init_cond.shape[0] != d_sys_.A.shape[0] or init_cond.shape[1] != 1:
            raise ValueError("wrong dimension of init condition")

    x = _cal_x(d_sys_.A, d_sys_.B, len(t[1:]), init_cond, u)
    y = []
    for i, x_k in enumerate(x):
        _ = d_sys_.C*x_k + d_sys_.D*u[i]
        y.append(_)
    if isinstance(sys_, SISO):
        y = [_[0, 0] for _ in y]
    else:
        y = [np.asarray(_).reshape(-1) for _ in y]
    return np.array(y), t


def _cal_x(G, H, n, x_0, u):
    """
    calculate x step by step
    """
    x = [x_0]
    for i in range(n):
        x_k = G*x[i] + H*u[i]
        x.append(x_k)
    return x


def step(sys_, t=None, *, plot=True):
    if t is None:
        t = np.linspace(0, 10, 1000)

    y, t = _any_input(sys_, t, 1)
    if plot:
        plot_response_curve(y, t, "step response")
    return y, t


def impulse(sys_, t=None, *, plot=True):
    if t is None:
        t = np.linspace(0, 10, 1000)

    u = np.zeros(t.shape)
    u[0] = len(t)/(t[-2] - t[0])  # It is a magic!!
    y, t = _any_input(sys_, t, u)
    y[0] = y[1]
    if plot:
        plot_response_curve(y, t, "impulse response")
    return y, t


def ramp(sys_, t=None, *, plot=True):
    if t is None:
        t = np.linspace(0, 10, 1000)

    u = t
    y, t = _any_input(sys_, t, u)
    if plot:
        plot_response_curve(y, t, "impulse response")
    return y, t


def any_input(sys_, t, input_signal=0, init_cond=None, *, plot=True):
    y, t = _any_input(sys_, t, input_signal, init_cond)
    if plot:
        plot_response_curve(y, t, "")


if __name__ == "__main__":
    import timeit

    r1 = timeit.timeit("tc.step(system, np.linspace(0, 10, 10000),plot=False)",
                       "import tcontrol as tc;import numpy as np;"
                       "system = tc.tf([5, 25, 30], [1, 6, 10, 8])", number=3)
    r2 = timeit.timeit("tc.impulse(system, np.linspace(0, 10, 10000),plot=False)",
                       "import tcontrol as tc;import numpy as np;"
                       "system = tc.tf([5, 25, 30], [1, 6, 10, 8])", number=3)
    r3 = timeit.timeit("tc.ramp(system, np.linspace(0, 10, 10000), plot=False)",
                       "import tcontrol as tc;import numpy as np;"
                       "system = tc.tf([5, 25, 30], [1, 6, 10, 8])", number=3)
    print(r1, r2, r3)
