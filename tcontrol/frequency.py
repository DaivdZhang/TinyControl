from .transferfunction import SISO, LinearTimeInvariant
import numpy as np
import math
from matplotlib import pyplot as plt

__all__ = ["nyquist"]


def nyquist(sys_, omega=None, *, plot=True):
    """

    :param sys_:
    :type sys_: SISO
    :param omega:
    :type omega: np.ndarray
    :param plot:
    :type plot: bool
    :return:
    :rtype:
    """
    if not isinstance(sys_, SISO):
        if isinstance(sys_, LinearTimeInvariant):
            raise NotImplementedError

    if omega is None:
        omega = np.linspace(0, 100, 10000)
    omega = 1j*omega

    num = np.poly1d(sys_.num)
    den = np.poly1d(sys_.den)

    result = num(omega)/den(omega)

    if plot:
        plt.axvline(x=0, color='black')
        plt.axhline(y=0, color='black')
        plt.plot(result.real, result.imag, '-', color='#069af3')
        plt.plot(result.real, -result.imag, '--', color='#fdaa48')

        arrow_pos = int(math.log(result.shape[0]))*2 + 1
        x1, x2 = np.real(result[arrow_pos]), np.real(result[arrow_pos + 1])
        y1, y2 = np.imag(result[arrow_pos]), np.imag(result[arrow_pos + 1])
        dx = x2 - x1
        dy = y2 - y1
        plt.arrow(x1, -y1, -dx, dy, head_width=0.04, color='#fdaa48')
        plt.arrow(x1, y1, dx, dy, head_width=0.04, color='#069af3')

        plt.scatter(-1, 0, s=30, color='r', marker='P')
        plt.grid()
        plt.title("Nyquist Plot")
        plt.xlabel('Real Axis')
        plt.ylabel('Imag Axis')
        plt.draw()

    return result, omega
