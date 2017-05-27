from src.lti import LinearTimeInvariant as LTI
from copy import deepcopy
import numpy as np
import sympy


class SISO(LTI):
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
                raise ValueError('1, 2 or 3 args expected. received %s' % length)

        else:
            raise ValueError('1, 2 or 3 args expected. received %s' % length)

        num = np.array(num)
        den = np.array(den)
        super().__init__(1, 1, dt)
        self.num = num
        self.den = den

    def __str__(self):
        s = sympy.Symbol('s')
        cs = 0
        rs = 0
        for i, n in enumerate(self.num[::-1]):
            cs += n * s**i
        for i, n in enumerate(self.den[::-1]):
            rs += n * s**i
        gs = cs / rs
        return gs.__str__()

    __repr__ = __str__

    def __neg__(self):
        num = deepcopy(self.num)
        num *= -1
        return SISO(num, self.den, self.dt)

    def __add__(self, other):
        """

        :param other:
        :type other: SISO
        :return:
        :rtype:
        """
        if self.dt == other.dt or (self.dt is not None and other.dt is None):
            dt = self.dt
        elif self.dt is None and other.dt is not None:
            dt = other.dt
        else:
            raise ValueError("")

        if all(self.den == other.den):
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
        :rtype:
        """
        num = np.convolve(self.num, other.num)
        den = np.convolve(self.den, other.den)

        s = sympy.Symbol('s')
        cs = 0
        rs = 0
        for i, n in enumerate(num[::-1]):
            cs += n * s**i
        for i, n in enumerate(den[::-1]):
            rs += n * s**i
        r = sympy.gcd(cs, rs)
        r = sympy.Poly(r)
        r = np.array(r.all_coeffs(), dtype=float)
        num = np.polydiv(num, r)[0]
        den = np.polydiv(den, r)[0]


        if self.dt == other.dt or (self.dt is not None and other.dt is None):
            dt = self.dt
        elif self.dt is None and other.dt is not None:
            dt = other.dt
        else:
            raise ValueError("")

        return SISO(num, den, dt)
