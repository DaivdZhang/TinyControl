from unittest import TestCase
from src.transferfunction import SISO
from src.rlocus import rlocus
from matplotlib import pyplot as plt


class TestRlocus(TestCase):
    def test_rlocus(self):
        system = SISO([1, 1], [1, 0.5, 1, 0])
        rlocus(system, xlim=[-4, 0.5])
        plt.show()
