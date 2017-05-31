import src
from matplotlib import pyplot as plt
from unittest import TestCase


class TestTimeResponse(TestCase):
    def test_step(self):
        system = src.tf([5, 25, 30], [1, 6, 10, 8])
        plt.figure(1)
        src.step(system)
        plt.show()

    def test_ramp(self):
        system = src.tf([5, 25, 30], [1, 6, 10, 8])
        plt.figure(2)
        src.ramp(system)
        plt.show()

    def test_impulse(self):
        system = src.tf([5, 25, 30], [1, 6, 10, 8])
        plt.figure(3)
        src.impulse(system)
        plt.show()
