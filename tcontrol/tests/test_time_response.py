import tcontrol
from matplotlib import pyplot as plt
from unittest import TestCase
import numpy as np


class TestTimeResponse(TestCase):
    def setUp(self):
        self.tf_ = tcontrol.tf([5, 25, 30], [1, 6, 10, 8])
        self.A = np.mat([[0, 1, 0], [0, 0, 1], [-8, -10, -6]])
        self.B = np.mat([[0], [0], [1]])
        self.C = np.mat([[30, 25, 5]])
        self.ss_ = tcontrol.ss(self.A.T, self.C.T, self.B.T, 0)

    def test_step(self):
        y1, t1 = tcontrol.step(self.tf_, plot=False)
        y2, t2 = tcontrol.step(self.ss_, plot=False)
        r = np.abs(y2 - y1)
        r = np.max(r)
        self.assertLessEqual(r, 1e-5)

    def test_ramp(self):
        y1, t1 = tcontrol.ramp(self.tf_, plot=False)
        y2, t2 = tcontrol.ramp(self.ss_, plot=False)
        r = np.abs(y2 - y1)
        r = np.max(r)
        self.assertLessEqual(r, 1e-5, r)

    def test_impulse(self):
        y1, t1 = tcontrol.impulse(self.tf_, plot=False)
        y2, t2 = tcontrol.impulse(self.ss_, plot=False)
        r = np.abs(y2 - y1)
        r = np.max(r)
        self.assertLessEqual(r, 1e-5, r)

    def test_any_input(self):
        t = np.linspace(0, 10, 1000)
        u = np.linspace(0, 10, 1000) + 1
        y1, t1 = tcontrol.any_input(self.tf_, t, u, plot=False)
        y2, t2 = tcontrol.any_input(self.ss_, t, u, plot=False)
        r = np.abs(y2 - y1)
        r = np.max(r)
        self.assertLessEqual(r, 1e-5, r)
