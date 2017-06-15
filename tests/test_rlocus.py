from unittest import TestCase
from tcontrol.transferfunction import SISO
from tcontrol.rlocus import rlocus
from matplotlib import pyplot as plt
import numpy as np


class TestRlocus(TestCase):
    def test_rlocus(self):
        system = SISO([0.5, 1], [0.5, 1, 1])
        rlocus(system, np.linspace(0, 100, 10000), xlim=[-5, 0.5])
        plt.show()
