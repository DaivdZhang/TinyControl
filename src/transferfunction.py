import src.lti
from copy import deepcopy
import numpy as np
import sympy

# TODO: trans function of MIMO
# TODO: improve error msg of class SISO
# TODO: implement zpk function
# TODO: simplify the trans function when calling __init__

__all__ = ["SISO", "tf"]


class SISO(src.lti.LinearTimeInvariant):
    def __init__(self, *args):
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
                raise ValueError("type of arg should be SISO, got %s".format(type(args[0])))

        else:
            raise ValueError('1, 2 or 3 arg(s) expected. received %s'.format(length))

        num = np.array(num)
        den = np.array(den)
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
        return SISO(num, self.den, self.dt)

    def __add__(self, other):
        """

        :param other:
        :type other: SISO
        :return:
        :rtype: SISO
        """
        dt = _get_dt(self, other)

        if np.array_equal(self.den, other.den):
            return SISO(np.polyadd(self.num, other.num), self.den, dt)

        den = np.convolve(self.den, other.den)
        num = np.polyadd(np.convolve(self.num, other.den), np.convolve(other.num, self.den))

        return SISO(num, den, dt)

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

        _, cs, rs = _siso_to_symbol(num, den)
        r = sympy.gcd(cs, rs)
        r = sympy.Poly(r)
        r = np.array(r.all_coeffs(), dtype=float)
        num = np.polydiv(num, r)[0]
        den = np.polydiv(den, r)[0]

        dt = _get_dt(self, other)

        return SISO(num, den, dt)

    __rmul__ = __mul__

    def pole(self):
        return np.roots(self.den)

    def zeros(self):
        return np.roots(self.num)

    def feedback(self, other=1, sign=1):
        if other == 1:
            other = SISO([1], [1], self.dt)

        dt = _get_dt(self, other)

        num = np.convolve(self.num, other.den)
        den = np.polyadd(np.convolve(self.num, other.num), np.convolve(self.den, other.den) * sign)

        return SISO(num, den, dt)


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
    s = sympy.Symbol('s')
    cs = 0
    rs = 0
    for i, n in enumerate(num[::-1]):
        cs += n * s**i
    for i, n in enumerate(den[::-1]):
        rs += n * s**i
    gs = cs / rs
    return gs, cs, rs


def tf(*args):
    length = len(args)
    if length == 2 or length == 3:
        return SISO(*args)
    elif length == 1:
        try:
            sys = SISO(*args)
            return sys
        except ValueError as e:
            print(e)
