from unittest import TestCase
from tcontrol.transferfunction import TransferFunction
from tcontrol.rlocus import rlocus
import numpy as np


class TestRlocus(TestCase):
    def test_rlocus(self):
        system = TransferFunction([0.5, 1], [0.5, 1, 1])
        r, _ = rlocus(system, xlim=[-5, 0.5])
        rlocus(TransferFunction([1], [1, 4, 3, 0]))
        rlocus(TransferFunction([1], [1, 2, 1, 0]))
        rlocus(TransferFunction([1, 3], [1, 6, 8, 0]))
