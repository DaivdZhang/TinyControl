import tcontrol
from matplotlib import pyplot as plt
from unittest import TestCase


class TestTimeResponse(TestCase):
    def test_step(self):
        system = tcontrol.tf([5, 25, 30], [1, 6, 10, 8])
        plt.figure(1)
        tcontrol.step(system)
        plt.show()

    def test_ramp(self):
        system = tcontrol.tf([5, 25, 30], [1, 6, 10, 8])
        plt.figure(2)
        tcontrol.ramp(system)
        plt.show()

    def test_impulse(self):
        system = tcontrol.tf([5, 25, 30], [1, 6, 10, 8])
        plt.figure(3)
        tcontrol.impulse(system)
        plt.show()
