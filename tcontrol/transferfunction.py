from tcontrol.lti import LinearTimeInvariant
from copy import deepcopy
import numpy as np
import sympy as sym

# TODO: trans function of MIMO
# TODO: improve error msg of class SISO

__all__ = ["SISO", "tf", "zpk", "ss2tf"]


class SISO(LinearTimeInvariant):
    def __init__(self, num, den, *, dt=None):
        num = np.array(np.poly1d(num))
        den = np.array(np.poly1d(den))
        num, den = _poly_simplify(num, den)

        super().__init__(1, 1, dt)
        self.num = num
        self.den = den

    def __str__(self):
        gs, _, _ = _siso_to_symbol(self.num, self.den)
        return gs.__str__()

    def __repr__(self):
        return '0x{0:x}: {1:s}'.format(id(self), self.__str__())

    def __eq__(self, other):
        if not isinstance(other, SISO):
            return False
        return np.array_equal(self.num, other.num) and np.array_equal(self.den, other.den)

    def __neg__(self):
        num = deepcopy(self.num)
        num *= -1
        return SISO(num, self.den, dt=self.dt)

    def __add__(self, other):
        """

        :param other:
        :type other: SISO
        :return:
        :rtype: SISO
        """
        dt = _get_dt(self, other)

        if np.array_equal(self.den, other.den):
            return SISO(np.polyadd(self.num, other.num), self.den, dt=dt)

        den = np.convolve(self.den, other.den)
        num = np.polyadd(np.convolve(self.num, other.den), np.convolve(other.num, self.den))

        return SISO(num, den, dt=dt)

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    __rsub__ = __sub__

    def __mul__(self, other):
        """

        :param other:
        :type other: SISO
        :return:
        :rtype: SISO
        """
        num = np.convolve(self.num, other.num)
        den = np.convolve(self.den, other.den)

        num, den = _poly_simplify(num, den)

        dt = _get_dt(self, other)

        return SISO(num, den, dt=dt)

    __rmul__ = __mul__

    def pole(self):
        return np.roots(self.den)

    def zero(self):
        return np.roots(self.num)

    def feedback(self, other=1, sign=1):
        """

        :param other: the transfer function of the feedback path
        :type other: SISO | int
        :param sign: if sign is 1 function will create the negative feedback.
                     otherwise the positive feedback
        :type sign: int
        :return:
        :rtype:
        """
        if other == 1:
            other = SISO([1], [1], dt=self.dt)

        dt = _get_dt(self, other)

        num = np.convolve(self.num, other.den)
        den = np.polyadd(np.convolve(self.num, other.num),
                         np.convolve(self.den, other.den)*sign)

        return SISO(num, den, dt=dt)


def _get_dt(sys1, sys2):
    """

    :param sys1:
    :type sys1: SISO
    :param sys2:
    :type sys2: SISO
    :return:
    :rtype:
    """
    if sys1.dt == sys2.dt or (sys1.dt is not None and sys2.dt is None):
        dt = sys1.dt
    elif sys1.dt is None and sys2.dt is not None:
        dt = sys2.dt
    else:
        raise ValueError(
            'Expected the same sampling time. got sys1:{0} sys2:{1}'.format(sys1.dt, sys2.dt))
    return dt


def _siso_to_symbol(num, den):
    s = sym.Symbol('s')
    cs = sym.Poly.from_list(num, gens=s)
    rs = sym.Poly.from_list(den, gens=s)
    gs = cs/rs
    return gs, cs, rs


def _poly_simplify(num, den):
    _, cs, rs = _siso_to_symbol(num, den)
    r = np.array(sym.gcd(cs, rs).as_poly().all_coeffs()).astype(num.dtype)
    if not np.array_equal(np.array([1]), r):
        num = np.polydiv(num, r)[0]
        den = np.polydiv(den, r)[0]

    return num, den


# def _poly_gcd(a, b):
#     s = sym.Symbol('s')
#     r = sym.gcd(a, b)
#     if r.is_Number:
#         return np.array([r], dtype=float)
#     p = sym.polys.polytools.poly(r)
#     n = 0
#     r = []
#     while True:
#         k = sym.polys.polytools.Poly.coeff_monomial(p, s**n)
#         if k.is_integer:
#             k = int(k)
#         elif k.is_real:
#             k = float(k)
#         elif k.is_complex:
#             k = complex(k)
#         else:
#             raise ValueError('unexpected coeff type')
#
#         if k == 0 and n != 0:
#             break
#         else:
#             r.insert(0, k)
#         n += 1
#     r = np.array(r)
#     return r


def tf(*args):
    """
    Use:
        Create a transfer function of a system

    Example:
        >>> from tcontrol import tf
        >>> system = tf([1, 1], [1, 0.5, 1])
        >>> print(system)
        (1.0*s + 1.0)/(1.0*s**2 + 0.5*s + 1.0)
        >>> system = tf(system)
        >>> print(system)
        (1.0*s + 1.0)/(1.0*s**2 + 0.5*s + 1.0)

    :param args:
    :type args: SISO | list[int | float]
    :return: the transfer function of the system
    :rtype: SISO
    """
    length = len(args)
    if length == 2:
        num, den = args
        dt = None
    elif length == 3:
        num, den, dt = args
    elif length == 1:
        if isinstance(args[0], SISO):
            num = args[0].num
            den = args[0].den
            dt = args[0].dt
        else:
            raise TypeError("type of arg should be SISO, got {0}".format(type(args[0])))

    else:
        raise ValueError('1, 2 or 3 arg(s) expected. received {0}'.format(length))
    sys_ = SISO(num, den, dt=dt)
    return sys_


def zpk(z, p, k):
    """
    Use:
        Create a transfer function by zeros, poles, and the gain k

    Example:
        >>> from tcontrol import zpk
        >>> system = zpk([], [1, 0.5, 1], 5.2)
        >>> print(system)
        5.2/(1.0*s**3 - 2.5*s**2 + 2.0*s - 0.5)


    :param z: zeros of a system
    :type z: np.ndarray | list | tuple
    :param p: poles of a system
    :type p: np.ndarray | list | tuple
    :param k: the gain of the system
    :type k: int | float
    :return: the transfer function of the system
    :rtype: SISO
    """
    num = np.array([k])
    for zi in z:
        num = np.convolve(num, np.array([1, -zi]))

    den = np.array([1])
    for pi in p:
        den = np.convolve(den, np.array([1, -pi]))
    return SISO(num, den)


def ss2tf(system):
    """

    :param system:
    :type system: StateSpace
    :return:
    :rtype:
    """
    T = system.to_controllable_form()
    A_ = T.I*system.A*T
    C_ = system.C*T
    a = np.asarray(A_[-1]).reshape(-1)*(-1)
    a = np.append(a, 1)
    a = a[::-1]
    b = np.asarray(C_[-1]).reshape(-1)
    return SISO(b[::-1], a)
