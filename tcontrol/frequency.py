from tcontrol.transferfunction import SISO, LinearTimeInvariant
import numpy as np
import math
from matplotlib import pyplot as plt

__all__ = ["nyquist", "bode"]


def nyquist(sys_, omega=None, *, plot=True):
    """

    :param sys_:
    :type sys_: SISO
    :param omega: values of angular frequency
    :type omega: np.ndarray
    :param plot:
    :type plot: bool
    :return:
    :rtype: (np.ndarray, np.ndarray)
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


def bode(sys_, omega=None, *, plot=True):
    """

    :param sys_:
    :type sys_: SISO
    :param omega: values of angular frequency
    :type omega: np.ndarray
    :param plot:
    :type plot: bool
    :return:
    :rtype: (np.ndarray, np.ndarray)
    """
    if not isinstance(sys_, SISO):
        if isinstance(sys_, LinearTimeInvariant):
            raise NotImplementedError

    if omega is None:
        omega = np.logspace(-1, 3)
    omega = omega*1j

    num = np.poly1d(sys_.num)
    den = np.poly1d(sys_.den)

    A = 20*np.log(np.abs(num(omega)/den(omega)))

    deg = np.zeros_like(omega)
    for i in sys_.zero():
        p = np.poly1d([1, -i])
        deg = deg + np.angle(p(omega), deg=True)
    for i in sys_.pole():
        p = np.poly1d([1, -i])
        deg = deg - np.angle(p(omega), deg=True)
    phi = deg

    if plot:
        plt.title("Bode Diagram")

        plt.subplot(2, 1, 1)
        plt.axvline(x=0, color='black')
        plt.axhline(y=0, color='black')
        plt.xscale('log')

        y_range = [i*20 for i in range(int(min(A))//20 - 1, int(max(A))//20 + 2)]
        plt.yticks(y_range)

        plt.plot(omega.imag, A, '-', color='#069af3')
        plt.grid(which='both')
        plt.ylabel('Magnitude/dB')

        plt.subplot(2, 1, 2)
        plt.axvline(x=0, color='black')
        plt.axhline(y=0, color='black')
        plt.xscale('log')

        y_range = [i*45 for i in range(int(min(A))//45 - 1, int(max(A))//45 + 2)]
        plt.yticks(y_range)

        plt.plot(omega.imag, phi, '-', color='#069af3')
        plt.grid(which='both')
        plt.ylabel('Phase/deg')
        plt.xlabel('Frequency/(rad/s)')

        plt.draw()

    return A, phi, omega
