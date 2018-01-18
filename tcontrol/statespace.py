import numpy as np
from tcontrol.lti import LinearTimeInvariant


class StateSpace(LinearTimeInvariant):
    """
    a class implement the state space model
    """

    def __init__(self, A, B, C, D, *, dt=None):
        if not isinstance(A, np.matrix):
            A = np.mat(A)
        if not isinstance(B, np.matrix):
            B = np.mat(B)
        if not isinstance(C, np.matrix):
            C = np.mat(C)
        if not isinstance(D, np.matrix):
            D = np.mat(D)

        # check shapes of matrix A B C D
        if A.shape[0] != A.shape[1]:
            raise ValueError(
                "{0} != {1}, wrong shape of A".format(A.shape[0], A.shape[1]))
        if B.shape[0] != A.shape[0]:
            raise ValueError(
                "{0} != {1}, wrong shape of B".format(B.shape[0], A.shape[0]))
        if C.shape[1] != A.shape[0]:
            raise ValueError(
                "{0} != {1}, wrong shape of C".format(C.shape[1], A.shape[0]))
        if D.shape[0] != C.shape[0] or D.shape[1] != B.shape[1]:
            raise ValueError(
                "{0} != ({1}, {2}), wrong shape of D".format(D.shape, C.shape[0],
                                                             B.shape[1]))

        super().__init__(B.shape[1], C.shape[0], dt)
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def __str__(self):
        a_str = 'A:\n' + str(self.A) + '\n\n'
        b_str = 'B:\n' + str(self.B) + '\n\n'
        c_str = 'C:\n' + str(self.C) + '\n\n'
        d_str = 'D:\n' + str(self.D) + '\n\n'
        return a_str + b_str + c_str + d_str

    __repr__ = __str__

    def __neg__(self):
        return StateSpace(self.A, self.B, -self.C, -self.D, dt=self.dt)

    def __add__(self, other):
        if self.D.shape != other.D.shape:
            raise ValueError(
                "shapes of D are not equal {0}, {1}".format(self.D.shape,
                                                            other.D.shape))

        if self.dt is None and other.dt is not None:
            dt = other.dt
        elif self.dt is not None and other.dt is None or self.dt == other.dt:
            dt = self.dt
        else:
            raise ValueError("different sampling times")

        A = np.zeros((self.A.shape[0] + other.A.shape[0],
                      self.A.shape[1] + other.A.shape[1]))
        A[0: self.A.shape[0], 0: self.A.shape[1]] = self.A
        A[self.A.shape[0]:, self.A.shape[1]:] = other.A
        B = np.concatenate((self.B, other.B), axis=0)
        C = np.concatenate((self.C, other.C), axis=1)
        D = self.D + other.D
        return StateSpace(A, B, C, D, dt=dt)

    __radd__ = __add__

    def __mul__(self, other):
        if not isinstance(other, StateSpace):
            other = _convert_to_ss(other)

        if self.outputs != other.inputs:
            raise ValueError("outputs are not equal to inputs")

        if self.dt is None and other.dt is not None:
            dt = other.dt
        elif self.dt is not None and other.dt is None or self.dt == other.dt:
            dt = self.dt
        else:
            raise ValueError("different sampling times")

        A = np.zeros((self.A.shape[0] + other.A.shape[0],
                      self.A.shape[1] + other.A.shape[1]))
        A[0: self.A.shape[0], 0: self.A.shape[1]] = self.A
        A[self.A.shape[0]:, self.A.shape[1]:] = other.A
        A[self.A.shape[0]:, 0: self.A.shape[0]] = other.B*self.C
        B = np.concatenate((self.B, other.B*self.D), axis=0)
        C = np.concatenate((other.D*self.C, other.C), axis=1)
        D = other.D*self.D
        return StateSpace(A, B, C, D, dt=dt)

    def feedback(self, K=1, sign=1):
        try:
            K = _convert_to_ss(K)
        except TypeError as e:
            raise TypeError from e
        A = self.A - self.B*K*self.C
        return StateSpace(A, self.B.copy(), self.C.copy(), np.mat([[0]]))

    def pole(self):
        """
        Usage:
            return the poles of the system

        :return: poles of the system
        :rtype: np.array
        """
        return np.linalg.eigvals(self.A)

    def controllability(self):
        """
        Usage:
            calculate and return the matrix [B A*B A^2*B ... A^(n-1)*B]

        :return: the matrix [B A*B A^2*B ... A^(n-1)*B]
        :rtype: np.matrix
        """
        tmp = self.B.copy()
        for i in range(1, self.A.shape[0]):
            tmp = np.concatenate((tmp, self.A**i*self.B), axis=1)
        return tmp

    def to_controllable_form(self):
        M = self.controllability().I
        p = np.asarray(M[-1]).reshape(-1)
        T = [np.asarray(p*self.A**i).reshape(-1) for i in range(self.A.shape[0])]
        T = np.mat(T)
        T = np.linalg.inv(T)
        return T

    def is_controllable(self):
        """
        Usage:
            the rank of the controllability matrix

        :return:
        :rtype: bool
        """
        if np.linalg.matrix_rank(self.controllability()) == self.A.shape[0]:
            return True
        else:
            return False

    def observability(self):
        """
        Usage:
            calculate and return the matrix [C
                                             C*A
                                             C*A^2
                                             ...
                                             C*A^(n-1)]

        :return: the matrix [C
                             C*A
                             C*A^2
                             ...
                             C*A^(n-1)]
        :rtype: np.matrix
        """
        tmp = self.C.copy()
        for i in range(1, self.A.shape[0]):
            tmp = np.concatenate((tmp, self.C*self.A**i), axis=0)
        return tmp

    def is_observable(self):
        """
        Usage:
            the rank of the observability matrix

        :return:
        :rtype: bool
        """
        if np.linalg.matrix_rank(self.observability()) == self.A.shape[0]:
            return True
        else:
            return False

    @classmethod
    def dual_system(cls, system):
        """

        :param system:
        :type system: StateSpace
        :return:
        :rtype: StateSpace
        """
        return cls(system.A.T.copy(), system.C.T.copy(), system.B.T.copy(), system.D.copy(),
                   dt=system.dt)

    @staticmethod
    def lyapunov(system):
        """

        :param system:
        :type system: StateSpace
        :return:
        :rtype:
        """
        np.eye(system.A.shape[0])


def ss(*args, **kwargs):
    length = len(args)
    if length == 1:
        _ss = args[0]
        A, B, C, D = _ss.A.copy(), _ss.B.copy(), _ss.C.copy(), _ss.D.copy()
    elif length == 4:
        A, B, C, D = args
    else:
        raise ValueError("1 or 4 args expected got {0}".format(length))
    dt = kwargs.get('dt')

    return StateSpace(A, B, C, D, dt=dt)


def tf2ss(*args):
    if len(args) == 1:
        num, den, dt = args[0].num, args[0].den, args[0].dt
    elif len(args) == 2:
        num, den = args
        dt = None
    else:
        raise TypeError
    num, den = np.poly1d(num), np.poly1d(den)
    if num.order > den.order:
        raise ValueError("wrong order num: {0} > den :{1}".format(num.order, den.order))

    if den[den.order] != 1:
        num /= den[den.order]
        den /= den[den.order]
    q, r = num/den
    num = r.coeffs
    bn = q.coeffs
    den = den.coeffs
    D = np.matrix(bn)

    # generate matrix A
    A = np.zeros((1, den.shape[0] - 2))
    A = np.concatenate((A, np.eye(den.shape[0] - 2)), axis=0)
    den = den[1:]
    den = np.mat(den[::-1])
    A = np.concatenate((A, -den.T), axis=1)

    B = np.zeros(A.shape[0])
    B[A.shape[0] - num.shape[0]:] = num
    B = np.mat(B[::-1]).T
    c = np.zeros(A.shape[0])
    c[-1] = 1
    C = np.mat(c)
    return StateSpace(A.T, C.T, B.T, D, dt=dt)


def _convert_to_ss(obj, **kwargs):
    if isinstance(obj, (float, int)):
        inputs = kwargs.get("inputs", 1)
        outputs = kwargs.get("outputs", 1)
        return StateSpace(np.matrix(0), np.zeros((1, inputs)), np.zeros(outputs, 1),
                          np.ones((outputs, inputs))*obj)
    elif isinstance(obj, LinearTimeInvariant):
        return tf2ss(obj)
    else:
        raise TypeError("wrong type. got {0}".format(type(obj)))
